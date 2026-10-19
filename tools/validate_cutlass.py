#!/usr/bin/env python3
"""thor_kda CLI validation bridge to the manifest-driven Graph/stream checker."""
import sys
from datetime import datetime,timezone
import os
from pathlib import Path
from smoke_gemm import main

if __name__=='__main__':
    if '--output' not in sys.argv:
        sys.argv.extend(['--output',str(Path(os.environ['KDA_THOR_ROOT'])/'results')+'/validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'])
    raise SystemExit(main())
