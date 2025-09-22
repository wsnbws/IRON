param (
    [string]$CONFIG,
    [string]$CHECKPOINT,
    [int]$GPUS,
    [int]$PORT = 29500
)

$PYTHONPATH = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "..") + ";$env:PYTHONPATH"

python -m torch.distributed.launch `
    --nproc_per_node=$GPUS `
    --master_port=$PORT `
    (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "test.py") `
    $CONFIG `
    $CHECKPOINT `
    --launcher pytorch `
    $args
