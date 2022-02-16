import aiohttp

from lib.Web.AsyncWebRequester import AsyncWebRequester


class ModelTrainerAsyncWebRequester(AsyncWebRequester):

    def __init__(self, url):
        super().__init__(url)

    async def post_to_submit_training_result(self, result_json):
        async with aiohttp.ClientSession() as session:
            try:
                await super().post_request(session, result_json)
            except Exception as e:
                raise e


