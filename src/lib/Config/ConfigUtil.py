
from lib.Config.ConfigAbc import ConfigAbc
from lib.Config.ConfigLocal import ConfigLocal as __ConfigLocal
from lib.Config.ConfigMongo import ConfigMongo as __ConfigMongo

__CONFIG_TYPE_MAP = {
    'local': __ConfigLocal,
    'mongo': __ConfigMongo,
}

__USE_CONFIG_TYPE = 'local'

ConfigUtil: ConfigAbc = __CONFIG_TYPE_MAP.get(__USE_CONFIG_TYPE)()