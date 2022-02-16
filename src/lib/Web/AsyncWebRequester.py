import asyncio

import aiohttp

from lib.Exceptions.AsyncWebException import ModelTrainerInternalProcessingException
from lib.Log.Log import Logger


class AsyncWebRequester:

    def __init__(self, url):
        self.url = url

    async def post_request(self, session, content):
        async with session.post(self.url, json=content) as response:
            resp = None
            try:
                resp = await response.read()

            except Exception as e:
                Logger.error('Failed at Post Request: %s', e)

            # if response status code is 500, which is internal server error
            if response.status == 500:
                raise ModelTrainerInternalProcessingException(
                    'Model Trainer internal processing exception [{}]'.format(content))

            return resp

