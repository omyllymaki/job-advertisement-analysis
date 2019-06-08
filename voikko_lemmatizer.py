import json
import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

URL = 'http://localhost:8000/lemmatize'


class VoikkoLemmatizer:

    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        """
        Takes tokens as input and returns base forms. If base form is not found, input token is returned instead.
        """
        data = {'tokens': tokens}
        result = requests.post(URL, json.dumps(data))
        base_forms_dict = json.loads(result.text)

        output_list = []
        for token in tokens:
            base_forms = base_forms_dict.get(token, None)
            if base_forms:
                output_list.append(base_forms[0])
            else:
                output_list.append(token)

        return output_list
