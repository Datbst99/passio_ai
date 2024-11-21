#!/usr/bin/env bash
PYTHON_VERSION=3.11
# Check for required dependencies
dependencies=("python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev")
missing_dependencies=()

dependencies=("python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev")
missing_dependencies=()

for dep in "${dependencies[@]}"; do
    if ! dpkg -s "$dep" &> /dev/null; then
        missing_dependencies+=("$dep")
    fi
done

if [ ${#missing_dependencies[@]} -gt 0 ]; then
    echo "Missing dependencies: ${missing_dependencies[*]}"
    echo "Please install them using 'sudo apt install ${missing_dependencies[*]}'"
    exit 1
fi

if python$PYTHON_VERSION --version &> /dev/null; then
    echo "Using Python version: $PYTHON_VERSION"
    if [ -f vendor/ok ]; then
        source vendor/bin/activate
    else
        echo "The environment is not ok. Running setup..."
        rm -rf vendor
        python$PYTHON_VERSION -m venv vendor  && \
        source vendor/bin/activate  && \
#        git clone https://github.com/coqui-ai/TTS  && \
        cd TTS  && \
        pip install --use-deprecated=legacy-resolver -e .  && \
        cd .. && \
        pip install -r requirements.txt && \
        python -m unidic download
        touch vendor/ok
    fi
else
    echo "Python version $PYTHON_VERSION is not installed. Please install it."
fi


