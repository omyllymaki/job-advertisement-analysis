import re
from abc import abstractmethod
from typing import List, Dict


def detect_features(text: str) -> Dict[str, List[str]]:
    detector_dict = {
        'programming_languages': ProgrammingLanguageDetector(),
        'technologies': TechnologyDetector(),
        'locations': LocationDetector(),
        'python_libraries': PythonLibraryDetector(),
    }

    results = {}
    for feature, detector in detector_dict.items():
        results[feature] = detector.detect(text)

    return results


def detect_technology_features(text: str) -> List[str]:
    detectors = [ProgrammingLanguageDetector(), TechnologyDetector(), PythonLibraryDetector()]

    results = []
    for detector in detectors:
        results += detector.detect(text)

    return results


class BaseDetector:

    @abstractmethod
    def detect(self, text: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def find_items_by_regexp(text: str, items: List[str]) -> List[str]:
        results = []
        for item in items:
            results += re.findall(f'(?:^|\W)({item})(?:$|\W)', text, re.IGNORECASE)
        return list(set(results))


class ProgrammingLanguageDetector(BaseDetector):
    PROGRAMMING_LANGUAGES = ['python', 'java', 'javascript', 'c', 'sql', 'ruby', 'matlab',
                             'node', 'sql', 'swift', 'typescript', 'php', 'kotlin',
                             'scala', 'haskell', 'html', 'js', 'css']

    def detect(self, text: str):
        return self.find_items_by_regexp(text, self.PROGRAMMING_LANGUAGES)


class TechnologyDetector(BaseDetector):
    TECHNOLOGIES = ['linux', 'windows', 'git', 'github', 'jenkins', 'docker', 'tableu',
                    'ansible', 'bitbucket', 'azure', 'google cloud', 'aws']

    def detect(self, text: str):
        return self.find_items_by_regexp(text, self.TECHNOLOGIES)


class LocationDetector(BaseDetector):
    LOCATIONS = ['helsinki', 'oulu', 'tampere', 'turku']

    def detect(self, text: str):
        return self.find_items_by_regexp(text, self.LOCATIONS)


class PythonLibraryDetector(BaseDetector):
    LIBRARIES = ['pandas', 'django', 'scikit', 'numpy', 'pytorch', 'tensorflow', 'keras',
                 'sqlalchemy', 'matplotlib', 'flask']

    def detect(self, text: str):
        return self.find_items_by_regexp(text, self.LIBRARIES)
