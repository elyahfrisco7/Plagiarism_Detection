créé ton environnement virtuel avec : python -m venv venv

Maintenant, pour activer cet environnement, la commande dépend du système d'exploitation que tu utilises. Voici les différentes options :

venv\Scripts\activate

Le code sql pour créer le bd et les et table :

CREATE DATABASE IF NOT EXISTS thesis_db;
USE thesis_db;

CREATE TABLE IF NOT EXISTS `theses` (
`id` INT AUTO_INCREMENT,
`title` VARCHAR(255),
`theme` VARCHAR(255),
`author` VARCHAR(255),
`university` VARCHAR(255),
`thesis_type` VARCHAR(50), -- "research" ou "professional"
`stage_location` VARCHAR(255), -- Lieu de stage ou d'étude
`methodology` TEXT, -- Méthodologie et objectifs
`results` TEXT, -- Résultats (technologies, outils, etc.)
`pdf_path` VARCHAR(255), -- Nom du fichier PDF
`theme_embedding` TEXT, -- Stockage embedding thème (JSON)
`stage_embedding` TEXT, -- Stockage embedding lieu (JSON)
`methodology_embedding` TEXT, -- Stockage embedding méthodologie (JSON)
`results_embedding` TEXT, -- Stockage embedding résultats (JSON)
`content_embedding` TEXT, -- Stockage embedding du contenu complet
`images_embedding` TEXT, -- Stockage embedding moyen des images
PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

installer cclip: Installation de CLIP depuis le dépôt GitHub d'OpenAI :

Le modèle CLIP n'est pas disponible en tant que package PyPI standard. Vous pouvez l'installer directement depuis son dépôt GitHub en utilisant la commande suivante :

pip install git+https://github.com/openai/CLIP.git
