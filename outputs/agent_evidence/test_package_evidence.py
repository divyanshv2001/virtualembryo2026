import unittest
from collections import Counter
from package_evidence import allowed,redact

class ExportBoundaries(unittest.TestCase):
    def test_hidden_reasoning_and_privileged_messages_excluded(self):
        for record in ({'type':'response_item','payload':{'type':'reasoning'}},
                       {'type':'response_item','payload':{'type':'message','role':'developer'}},
                       {'type':'response_item','payload':{'type':'message','role':'assistant','channel':'analysis'}},
                       {'type':'world_state','payload':{}},
                       {'type':'compacted','payload':{}}):
            self.assertFalse(allowed(record))
    def test_recorded_tool_and_user_events_retained(self):
        self.assertTrue(allowed({'type':'response_item','payload':{'type':'function_call_output'}}))
        self.assertTrue(allowed({'type':'event_msg','payload':{'type':'user_message'}}))
    def test_nested_credentials_redacted_in_place(self):
        dummy='apikey_'+'a'*32+'_'+'b'*64
        escaped=dummy.replace('_','\\_')
        before={'timestamp':'original','payload':['local-secret',dummy,escaped,'result']}
        count=Counter();after=redact(before,'local-secret',count)
        self.assertEqual(after['timestamp'],'original')
        self.assertEqual(after['payload'][-1],'result')
        self.assertEqual(after['payload'][:3],['[REDACTED_AUTHENTICATION_CREDENTIAL]']*3)
        self.assertEqual(count['exact_credential_replacements'],1)
        self.assertEqual(count['credential_pattern_replacements'],2)

if __name__=='__main__':unittest.main()
