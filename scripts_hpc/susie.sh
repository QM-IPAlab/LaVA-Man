export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export CLIPORT_ROOT=$(pwd)

python -m cliport.test_susie