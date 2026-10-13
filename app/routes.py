# app/routes.py
import os
import json
import tempfile
import numpy as np
from flask import request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from app.models import Thesis, SessionLocal
from app.config import UPLOAD_FOLDER, EMBEDDING_DIM, get_sentence_model, get_collection
from app.utils import extract_text_from_pdf, extract_images_from_pdf, get_image_embedding
from app.controllers import compute_similarity_breakdown
from app.ai_extractor import extract_metadata_with_ai, AIExtractionError
from app.weights import CRITERIA, CRITERIA_LABELS, CRITERIA_ICONS, get_weights, save_weights, reset_weights, is_default

def register_routes(app):
    """Register all Flask routes"""
    
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        """
        Récupère le PDF et les champs du formulaire,
        extrait le texte + images, calcule les embeddings,
        stocke dans MySQL, puis indexe le contenu dans ChromaDB.
        """
        title = request.form.get('title')
        theme = request.form.get('theme')
        author = request.form.get('author')
        university = request.form.get('university')
        thesis_type = request.form.get('thesis_type')
        stage_location = request.form.get('stage_location')
        methodology = request.form.get('methodology')
        results_text = request.form.get('results')
        file = request.files.get('pdf_file')

        if not file:
            return "Aucun fichier fourni", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extraction
        full_text = extract_text_from_pdf(file_path)
        images = extract_images_from_pdf(file_path)

        # Get models
        model = get_sentence_model()
        
        # Embeddings texte
        theme_emb = model.encode(theme).tolist()
        stage_emb = model.encode(stage_location).tolist()
        meth_emb = model.encode(methodology).tolist()
        results_emb = model.encode(results_text).tolist()
        full_content_emb = model.encode(full_text).tolist()

        # Embeddings images (moyenne si plusieurs)
        image_embeddings = []
        for img in images:
            emb = get_image_embedding(img)
            image_embeddings.append(emb)

        if image_embeddings:
            images_emb = np.mean(np.array(image_embeddings), axis=0).tolist()
        else:
            images_emb = [0.0] * EMBEDDING_DIM

        # Stockage MySQL
        session = SessionLocal()
        thesis = Thesis(
            title=title,
            theme=theme,
            author=author,
            university=university,
            thesis_type=thesis_type,
            stage_location=stage_location,
            methodology=methodology,
            results=results_text,
            pdf_path=filename,
            theme_embedding=json.dumps(theme_emb),
            stage_embedding=json.dumps(stage_emb),
            methodology_embedding=json.dumps(meth_emb),
            results_embedding=json.dumps(results_emb),
            content_embedding=json.dumps(full_content_emb),
            images_embedding=json.dumps(images_emb)
        )
        session.add(thesis)
        session.commit()
        thesis_id = thesis.id
        session.close()

        # Ajout dans la collection ChromaDB
        collection = get_collection()
        collection.add(
            ids=[str(thesis_id)],
            embeddings=[full_content_emb],
            metadatas=[{
                "title": title,
                "theme": theme,
                "author": author,
                "university": university,
                "thesis_type": thesis_type,
                "stage_location": stage_location
            }]
        )

        return redirect(url_for('index'))

    @app.route('/settings', methods=['GET'])
    def settings():
        """
        Page de configuration de la pondération des 6 critères de similarité
        (Thème, Lieu de stage, Méthodologie, Résultats, Contenu, Images).
        Par défaut : poids égal (1/6 chacun) ; modifiable et persisté dans
        weights_config.json (voir app/weights.py).
        """
        return render_template(
            'settings.html',
            criteria=CRITERIA,
            labels=CRITERIA_LABELS,
            crit_icons=CRITERIA_ICONS,
            weights=get_weights(),
            is_default=is_default(),
            saved=request.args.get('saved'),
            reset=request.args.get('reset'),
        )

    @app.route('/settings', methods=['POST'])
    def settings_save():
        action = request.form.get('action')
        if action == 'reset':
            reset_weights()
            return redirect(url_for('settings', reset=1))

        raw_weights = {c: request.form.get(c, 0) for c in CRITERIA}
        save_weights(raw_weights)
        return redirect(url_for('settings', saved=1))

    @app.route('/api/extract-metadata', methods=['POST'])
    def api_extract_metadata():
        """
        Reçoit un PDF, en extrait le texte (sans le stocker définitivement),
        puis demande à Claude Haiku 4.5 de déduire les champs de la fiche
        technique (titre, thème, auteur, université, type, lieu, méthodologie,
        résultats). Utilisé pour l'auto-remplissage du formulaire d'upload.
        """
        file = request.files.get('pdf_file')
        if not file:
            return jsonify({"error": "Aucun fichier fourni"}), 400

        tmp_path = None
        try:
            suffix = os.path.splitext(secure_filename(file.filename or ""))[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp)
                tmp_path = tmp.name

            full_text = extract_text_from_pdf(tmp_path)
            metadata = extract_metadata_with_ai(full_text)
            return jsonify(metadata.model_dump())
        except AIExtractionError as e:
            return jsonify({"error": str(e)}), 502
        except Exception as e:
            return jsonify({"error": f"Erreur inattendue : {e}"}), 500
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.route('/compare', methods=['GET'])
    def compare():
        """
        Compare deux mémoires par ID -> page HTML élégante (ou JSON si demandé).
        Ex: /compare?id1=1&id2=2
        """
        id1 = request.args.get('id1')
        id2 = request.args.get('id2')
        if not id1 or not id2:
            return "Les deux IDs sont requis pour la comparaison", 400

        session = SessionLocal()
        thesis1 = session.query(Thesis).filter(Thesis.id == int(id1)).first()
        thesis2 = session.query(Thesis).filter(Thesis.id == int(id2)).first()
        session.close()

        if not thesis1 or not thesis2:
            return "Un ou plusieurs documents n'ont pas été trouvés", 404

        sims, overall = compute_similarity_breakdown(thesis1, thesis2)

        # JSON si explicitement demandé
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({
                "thesis1": thesis1.title,
                "thesis2": thesis2.title,
                "similarity_percentage": overall,
                "breakdown": {k: float(v) for k, v in sims.items()}
            })

        # Sinon, rendu HTML
        return render_template(
            "compare.html",
            thesis1=thesis1,
            thesis2=thesis2,
            overall=overall,
            sims=sims
        )

    @app.route('/search', methods=['GET'])
    def search():
        """
        Recherche sémantique -> calcule l'embedding de la requête,
        interroge ChromaDB pour trouver jusqu'à 100 documents proches.
        Retourne la page 'results.html' avec la liste.
        """
        query = request.args.get('query', '')
        if not query:
            return "Une requête est nécessaire", 400

        print(f"DEBUG: Search query received: '{query}'")
        
        model = get_sentence_model()
        query_embedding = model.encode(query).tolist()

        collection = get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=100,
            include=["metadatas", "distances"]
        )
        
        print(f"DEBUG: ChromaDB returned {len(results['ids'][0])} results")

        session = SessionLocal()
        theses_found = []
        for idx, thesis_id in enumerate(results['ids'][0]):
            t = session.query(Thesis).filter(Thesis.id == int(thesis_id)).first()
            if t:
                content_score = max(0, 100 - results['distances'][0][idx] * 100)
                theses_found.append({
                    "id": t.id,
                    "title": t.title,
                    "theme": t.theme,
                    "author": t.author,
                    "university": t.university,
                    "thesis_type": t.thesis_type,
                    "stage_location": t.stage_location,
                    "methodology": t.methodology,
                    "results": t.results,
                    "content_score": content_score,
                    "pdf_path": t.pdf_path
                })
        session.close()
        
        print(f"DEBUG: Found {len(theses_found)} theses in MySQL matching ChromaDB IDs")

        return render_template('results.html', results=theses_found)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_FOLDER, filename)

    @app.route('/theses')
    def theses():
        session = SessionLocal()
        items = session.query(Thesis).order_by(Thesis.id.desc()).all()
        session.close()
        return render_template('theses.html', theses=items)

    @app.route('/api/theses/<int:thesis_id>', methods=['DELETE'])
    def delete_thesis(thesis_id):
        """
        Supprime définitivement un mémoire : la ligne MySQL, son entrée dans
        l'index ChromaDB (recherche sémantique) et le fichier PDF sur disque.
        """
        session = SessionLocal()
        thesis = session.query(Thesis).filter(Thesis.id == thesis_id).first()
        if not thesis:
            session.close()
            return jsonify({"error": "Mémoire introuvable"}), 404

        pdf_path = thesis.pdf_path
        session.delete(thesis)
        session.commit()
        session.close()

        # Retire l'entrée de ChromaDB (ne doit pas faire échouer la suppression
        # si l'index ne la contient pas / est indisponible).
        try:
            get_collection().delete(ids=[str(thesis_id)])
        except Exception as e:
            print(f"Avertissement : suppression ChromaDB échouée pour l'id {thesis_id}: {e}")

        # Supprime le fichier PDF associé.
        if pdf_path:
            file_path = os.path.join(UPLOAD_FOLDER, pdf_path)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Avertissement : suppression du fichier {file_path} échouée : {e}")

        return jsonify({"success": True})

    @app.route('/api/scan', methods=['GET'])
    def api_scan():
        target_id = request.args.get('id')
        if not target_id:
            return jsonify({"error": "Paramètre id manquant"}), 400

        session = SessionLocal()
        all_theses = session.query(Thesis).all()
        target = None
        for t in all_theses:
            if str(t.id) == str(target_id):
                target = t
                break

        if target is None:
            session.close()
            return jsonify({"error": "Mémoire introuvable"}), 404

        results = []
        for t in all_theses:
            if t.id == target.id:
                continue
            sims, overall = compute_similarity_breakdown(target, t)
            results.append({
                "id": t.id,
                "title": t.title,
                "author": t.author,
                "university": t.university,
                "thesis_type": t.thesis_type,
                "stage_location": t.stage_location,
                "pdf_path": t.pdf_path,
                "overall": float(overall),
                "sims": {k: float(v) for k, v in sims.items()},
                "compare_url": url_for('compare', id1=target.id, id2=t.id)
            })

        session.close()

        # Tri décroissant par score global
        results.sort(key=lambda x: x["overall"], reverse=True)

        return jsonify({
            "target": {
                "id": target.id,
                "title": target.title,
                "author": target.author,
                "university": target.university,
                "thesis_type": target.thesis_type,
                "stage_location": target.stage_location,
                "pdf_path": target.pdf_path
            },
            "results": results
        })
