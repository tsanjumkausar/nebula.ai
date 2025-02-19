import pytest
import nebula as nb

# Fixture to initialize the NebulaFunc class for each test
@pytest.fixture
def classFunc():
    return nb.NebulaFunc()

class TestNebulaPossTest:
    # Test case for negative sentiment analysis
    def test_neg_test_1(self, classFunc):
        result = classFunc.sentimentAn("i feel so bad about myself!!")
        assert result == "The user seems to be sad."

    # Test case for positive sentiment analysis
    def test_pos_test_1(self, classFunc):
        result = classFunc.sentimentAn("i feel so fucking happy about myself!!")
        assert result == "The user seems to be happy!"

    # Additional test cases for negative sentiment
    def test_neg_test_2(self, classFunc):
        result = classFunc.sentimentAn("I'm really upset today.")
        assert result == "The user seems to be sad."

    def test_neg_test_3(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really down.")
        assert result == "The user seems to be sad."

    def test_neg_test_4(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really low.")
        assert result == "The user seems to be sad."

    def test_neg_test_5(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really down.")
        assert result == "The user seems to be sad."

    # Additional test cases for positive sentiment
    def test_pos_test_2(self, classFunc):
        result = classFunc.sentimentAn("I'm so excited about my new job!")
        assert result == "The user seems to be happy!"

    def test_pos_test_3(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really upbeat.")
        assert result == "The user seems to be happy!"

    def test_pos_test_4(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really high.")
        assert result == "The user seems to be happy!"

    def test_pos_test_5(self, classFunc):
        result = classFunc.sentimentAn("I'm feeling really upbeat.")
        assert result == "The user seems to be happy!"
