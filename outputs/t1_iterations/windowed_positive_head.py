"""Past-stage window sensitivity for conditional positive gene decoders."""
import numpy as np
from ridge_conditional_head import RidgeConditionalPositiveForecast


class PastRowsView:
    """Map local row indices lazily; never materialize the full gene panel."""
    def __init__(self,x,rows):self.x=x;self.rows=np.asarray(rows);self.shape=(len(rows),x.shape[1])
    def __getitem__(self,key):
        if isinstance(key,tuple):return self.x[self.rows[key[0]],key[1]]
        return self.x[self.rows[key]]


class WindowedPositiveForecast(RidgeConditionalPositiveForecast):
    def __init__(self,x,stages,cutoff,donors,panel,symbols,net,center,scale,features,head_stages=None):
        if head_stages not in [None,1,2,3]:raise ValueError('Undeclared decoder window')
        rows=np.flatnonzero(stages<=cutoff)
        if head_stages is not None:
            selected=np.unique(stages[rows])[-head_stages:]
            rows=rows[np.isin(stages[rows],selected)]
        if not len(rows):raise ValueError('No permitted decoder cells')
        # Subset actual rows and their real stage labels; the frozen encoder
        # still represents all past stages. No invented or relabelled times.
        super().__init__(PastRowsView(x,rows),stages[rows],cutoff,donors,panel,symbols,net,center,scale,features,ridge=1.)
        self.audit.update(decoder_window_stages=head_stages,decoder_fit_stages=np.unique(stages[rows]).tolist(),
            scope_decoder='Restrict only the positive/detection head fit rows and resulting centering/support. Frozen representation, drift, forecast strength, guards and scorer unchanged. Detection is not used in abundance forecasts.')
