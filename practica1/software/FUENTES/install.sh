echo "Creating virtual environment..."
python3 -m venv env
echo "Activating virtual environment..."
source ./env/bin/activate
echo "Installing dependencies..."
sleep 1
pip3 install -r requirements.txt
