class CFG:
    def __init__(self, cfg: dict) -> None:
        for k, v in cfg.items():
            setattr(self, k, v)
    
    