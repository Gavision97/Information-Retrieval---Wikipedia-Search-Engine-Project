import os
from pyngrok import ngrok
import search_frontend as se
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "irprojectilayvictor-e5016b01bf5b.json"
ngrok.set_auth_token("2cm7N9F92W8iDy4CD5r3EQgRi0B_2nNi4EgiEVF73siGHaaYt")
public_url = ngrok.connect(5000).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
# Update any base URLs to use the public ngrok URL
se.app.config["BASE_URL"] = public_url
se.app.run()