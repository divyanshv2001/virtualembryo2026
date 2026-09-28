import unittest
from monte_carlo_stability import summarize


class StabilityGateTests(unittest.TestCase):
    def row(self, score):
        return {'calibration_valid':True, 'local_score':score, 'persistence_score':50.,
            'skills':{'de_score':.8, 'de_direction':.8, 'mmd_u':.8, 'variogram':.8}}

    def test_lucky_maximum_and_short_pilot_cannot_pass(self):
        self.assertFalse(summarize([self.row(80.)]*16, True)['final_gate_passed'])
        result = summarize([self.row(60.)]*63+[self.row(99.)], True)
        self.assertEqual(result['maximum_score'], 99.)
        self.assertFalse(result['final_gate_passed'])

    def test_invalid_calibration_cannot_be_dropped(self):
        rows = [self.row(80.)]*64+[{'calibration_valid':False}]
        result = summarize(rows, True)
        self.assertFalse(result['final_gate_passed']); self.assertIsNone(result['mean_score'])
        self.assertEqual(result['invalid_replicates'], 1)


if __name__ == '__main__': unittest.main()
