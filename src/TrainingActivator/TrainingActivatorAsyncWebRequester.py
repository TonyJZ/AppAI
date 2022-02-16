import asyncio
import aiohttp

from lib.Exceptions.AsyncWebException import ModelTrainerInternalProcessingException
from lib.Log.Log import Logger
from lib.Web.AsyncWebRequester import AsyncWebRequester


class TrainingActivatorAsyncWebRequester(AsyncWebRequester):

    def __init__(self, url):
        super().__init__(url)

    async def post_to_model_trainer(self, type_ids, pipeline):
        tasks = []
        async with aiohttp.ClientSession() as session:
            try:
                for type_id in type_ids:
                    content = {'type_id': type_id, 'pipeline_name': pipeline}
                    task = asyncio.ensure_future(super().post_request(session, content))
                    tasks.append(task)
                    Logger.info('Scheduler activated training event: %s', content)

                results = await asyncio.gather(*tasks)
                Logger.debug(results)

            except Exception as e:
                Logger.error('Exception happened while waiting for response. %s', e)

    async def post_to_model_trainer_from_request(self, type_id, pipeline, stime=None, etime=None):
        async with aiohttp.ClientSession() as session:
            try:

                content = {'type_id': type_id, 'pipeline_name': pipeline}
                if stime is not None and etime is not None:
                    content.setdefault('s_time', stime)
                    content.setdefault('e_time', etime)

                Logger.debug('User Request activated training event: %s', content)
                result = await super().post_request(session, content)
                Logger.debug(result)
                return result

            except Exception as e:
                Logger.error('Exception happened while waiting for response. %s', e)
                raise e

