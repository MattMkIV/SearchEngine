from whoosh import index
from abc import ABC, abstractmethod
from spellchecker import SpellChecker
from whoosh.qparser import QueryParser
from sentimentRanking import SentimentRanking


class BaseSearcher(ABC):
    """Base abstract class for an Index Searcher"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def search(self):
        pass


class SentimentSearcher(BaseSearcher):
    """Class that searches the index based on the list of query parsed by the InputCleaner"""
    def __init__(self, indexDir, tokenInput, queryList, sentiment, sentimentType):
        """
        :param indexDir: Directory which contains the index
        :param tokenInput: Tokenized user query
        :param queryList: List of formatted query for the index
        :param sentiment: True if the user wants to filter by sentiment
        :param sentimentType: Contains the sentiment type the user is looking for (e.g.: 'positive')
        """
        # The attributes are protected because they need to be accessed by classes of a lower hierarchy
        self.__correctedTokenInput = []
        self.__misspelled = None
        self.__spell = None
        self._corrector = None
        self._finalResult = None
        self._resultSentiment = None
        self._indexDir = indexDir
        self._ix = index.open_dir(indexDir)
        self._searcher = self._ix.searcher()
        self._parser = QueryParser(sentimentType, self._ix.schema)
        self._queryList = queryList
        self._sentiment = sentiment
        self._tokenInput = tokenInput
        self._sentimentType = sentimentType
        self.__reader = self._ix.reader()
        self._tfIdfRanking = False

    def search(self):
        """Searches the index with the queries provided in queryList"""
        self._finalResult = self._searcher.search(self._parser.parse(self._queryList[0]), limit=None)
        self._finalResult.extend(self._searcher.search(self._parser.parse(self._queryList[1]), limit=None))
        self._finalResult.extend(self._searcher.search(self._parser.parse(self._queryList[2]), limit=None))

        if self._sentiment:
            self._resultSentiment = self._searcher.search(self._parser.parse(self._queryList[3]))

            if self._resultSentiment.is_empty():  # The function filter doesn't actually filter if resultSentiment is empty
                self._finalResult = self._resultSentiment
            else:
                self._finalResult.filter(self._resultSentiment)  # intersection

    def corrector(self):
        foundCorrection = False
        self.__spell = SpellChecker()

        for word in self._tokenInput:
            correctedWord = self.__spell.correction(word)

            if correctedWord is not None:
                byteStringCorrectedWord = correctedWord.encode()
            else:
                return False

            if byteStringCorrectedWord in self.__reader.lexicon('originalProductTitle') \
                    or byteStringCorrectedWord in self.__reader.lexicon('originalReviewTitle') \
                    or byteStringCorrectedWord in self.__reader.lexicon('originalReviewContent'):
                foundCorrection = True
                self.__correctedTokenInput.append(correctedWord)
            else:
                self.__correctedTokenInput.append(word)

        if foundCorrection:
            return ' '.join(self.__correctedTokenInput)
        else:
            return False


class SentimentSearcherRanker(SentimentSearcher):
    """Class that extends SentimentSearcher by also adding a ranking function"""
    def __init__(self, indexDir, tokenInput, queryList, sentiment, sentimentType):
        self.__ranker = None
        self.__resultList = None
        super().__init__(indexDir, tokenInput, queryList, sentiment, sentimentType)

    def search(self):
        super().search()

    def ranking(self):
        """
        Ranks the result based on how much a document is relevant to the user query
        :return: List of ordered results by relevancy
        """
        self.__ranker = SentimentRanking(self._finalResult, self._tokenInput, self._sentiment, self._sentimentType)

        self.__resultList = self.__ranker.calculateRank()
        return self.__resultList
