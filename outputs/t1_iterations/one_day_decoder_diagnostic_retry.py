"""Distinct diagnostic retry, preserves original failed code/report/events."""
import torch
from threadpoolctl import threadpool_limits
import one_day_decoder_diagnostic as diagnostic

diagnostic.RUN = diagnostic.HERE / 'private/one_day_decoder_diagnostic_retry_01'
diagnostic.PUBLIC = diagnostic.HERE / 'ONE_DAY_DECODER_DIAGNOSTIC_RETRY_RESULTS.json'

if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        diagnostic.main()
