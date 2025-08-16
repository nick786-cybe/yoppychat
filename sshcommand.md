# Update your server's package list
sudo apt update

# Install Python, pip, virtual environment tools, and the Nginx web server
sudo apt install python3 python3-pip python3-venv nginx -y

# Clone your repository (replace with your actual URL)
git clone https://github.com/your-username/your-repository-name.git

# Navigate into the new project directory
cd your-repository-name

# Create a virtual environment named "venv"
python3 -m venv venv

# Activate the virtual environment (you must do this every time you log in)
source venv/bin/activate

# Install all the required Python libraries from your requirements file
pip install -r requirements.txt

# Install the gevent library for the Gunicorn worker
pip install gevent

# Create and edit your environment variables file
nano .env

# Create and edit the supervisor service file for Gunicorn and Huey
sudo nano /etc/supervisor/conf.d/yoppychat.conf

# Create and edit the Nginx configuration file
sudo nano /etc/nginx/sites-available/yoppychat

# Link the Nginx config to enable it
sudo ln -s /etc/nginx/sites-available/yoppychat /etc/nginx/sites-enabled

# Test your Nginx configuration for syntax errors
sudo nginx -t

# Restart Nginx to apply changes
sudo systemctl restart nginx

# Tell Supervisor to read the new/changed config files
sudo supervisorctl reread

# Tell Supervisor to enact any new configurations
sudo supervisorctl update

# Restart all services managed by Supervisor (Gunicorn and Huey)
sudo supervisorctl restart all

# View the last 50 lines of your Gunicorn error log in real-time
sudo tail -f /var/log/creatorchat-gunicorn.err.log

# Run the one-time script to clean up orphaned embeddings
python3 cleanup_embeddings.py