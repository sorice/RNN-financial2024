import json
import logging

from pydantic import BaseModel

from config import get_config

config = get_config()


class ModelEncoder(json.JSONEncoder):
    """Clase personalizada para codificar objetos en formato JSON."""

    def default(self, obj) -> dict:
        """Método al que se llama cuando se encuentra un objeto no serializable en la codificación JSON.

        :param obj: Objeto a codificar.
        :return Cualquiera: Representación JSON del resultado del objeto del método de clase base 'predeterminado'.
        """

        if isinstance(obj, BaseModel):
            return obj.dict()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


# logger object with the name of the current module (__name__) is obtained.
logger = logging.getLogger(__name__)
logger.setLevel(config.debug_level)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(config.debug_level)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)