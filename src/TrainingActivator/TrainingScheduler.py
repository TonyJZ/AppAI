import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from lib.Log.Log import Logger
from src.TrainingActivator.TrainingActivatorConfig import TrainingActivatorConfig


class TrainingScheduler:
    scheduler = None

    def __init__(self, requester):
        # singleton scheduler
        if TrainingScheduler.scheduler is None:
            Logger.debug("Initializing the singleton scheduler")
            TrainingScheduler.scheduler = AsyncIOScheduler()

        self.scheduler = TrainingScheduler.scheduler
        self.requester = requester

    def start(self, loop):
        if not self.scheduler.running:
            self.scheduler.start()

        Logger.info('Starting the Scheduler, jobs: %s', self.get_jobs())
        try:
            loop.run_forever()

        except (KeyboardInterrupt, SystemExit):
            Logger.debug('Exiting')

    def shutdown(self):
        Logger.info('Shutting down the Scheduler')
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown()

    def add_job(self, pipeline, schedule_config):

        type_ids = schedule_config.get('type_ids')
        schedule_type = schedule_config.get('schedule_type')
        schedule = schedule_config.get('schedule')
        self.scheduler.add_job(self.requester.post_to_model_trainer, schedule_type,
                               kwargs={'type_ids': type_ids, 'pipeline': pipeline},
                               id=pipeline, **schedule)
        Logger.debug(self.get_job(pipeline))

    def remove_job(self, pipeline):
        self.scheduler.remove_job(pipeline)

    def get_job(self, pipeline):
        job = self.scheduler.get_job(pipeline)
        job_info = {
            'pipeline_name': job.id,
            'trigger': job.trigger
        }

        return job_info

    def get_jobs(self):
        # self.scheduler.print_jobs()
        jobs = self.scheduler.get_jobs()
        job_list = []
        for job in jobs:
            job_info = {
                'pipeline_name': job.id,
                'trigger': job.trigger,
                'next_fire': job.next_run_time
            }
            job_list.append(job_info)

        return job_list


def training_scheduler_process(requester):
    # setup event loop for subprocess
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    loop = asyncio.get_event_loop()

    # initializing the requester
    scheduler = TrainingScheduler(requester)

    # configure pipeline scheduling
    for pipeline_config in TrainingActivatorConfig.pipelines:
        scheduler.add_job(pipeline_config.get('pipeline_name'), pipeline_config)

    # start the scheduler
    scheduler.start(loop)

    scheduler.shutdown()
    loop.close()

    Logger.info('Training Scheduler Process is terminating')
