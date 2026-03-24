from yarp.reaction.external.calc_base import AsyncYarpCalculator

class IRCValTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not getattr(self.rxn, 'ts_geom', None) or not self.rxn.ts_geom.get("optimized"):
            return False
        return True