"""Meaningful cache/schema checks; no network or scientific data."""
import unittest
from compact_jev import cache_key,validate_answers

class Contracts(unittest.TestCase):
    def test_missing_answers_never_become_authoritative(self):
        with self.assertRaises(ValueError):
            validate_answers({'gate':{'type':'noul'}},{})
    def test_invalid_and_boolean_probabilities_rejected(self):
        for value in (float('nan'),float('inf'),-1,2,True):
            with self.assertRaises(ValueError):
                validate_answers({'gate':{'type':'noul'}},{'gate':{'noul':value}})
    def test_choice_and_confidence_must_be_valid(self):
        q={'route':{'type':'choice','criteria':{'expert':'Review'}}}
        for a in ({'choice':'other','confidence':1},{'choice':'expert'},{'choice':'expert','confidence':True}):
            with self.assertRaises(ValueError):
                validate_answers(q,{'route':a})
        validate_answers(q,{'route':{'choice':'expert','confidence':.7}})
    def test_cache_changes_with_semantics_not_dictionary_order(self):
        self.assertEqual(cache_key({'a':1,'b':2}),cache_key({'b':2,'a':1}))
        for changed in ({'a':2,'b':2},{'model':'new','a':1,'b':2}):
            self.assertNotEqual(cache_key({'a':1,'b':2}),cache_key(changed))

if __name__=='__main__':
    unittest.main()
