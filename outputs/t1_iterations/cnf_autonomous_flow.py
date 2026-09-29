"""Calendar-time independent velocity with unchanged CNF initialization."""
from cnf_density_flow import DensityFlowNet


class AutonomousDensityFlowNet(DensityFlowNet):
    def velocity(self,time,z):
        # Zero inactive time channel preserves parameter shapes and RNG lineage.
        return super().velocity(0.,z)
