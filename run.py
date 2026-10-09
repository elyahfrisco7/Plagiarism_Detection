# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    # threaded=True : évite qu'une requête lente (ex. l'appel à l'IA pour
    # l'auto-remplissage, ~5-15s) ne bloque les autres requêtes du serveur de
    # développement (chargement de page, assets statiques, etc.).
    app.run(debug=True, threaded=True)
