# =============================================================================
# Dockerfile Frontend - AestheticEngine (Vue 3 + Vite + TypeScript + Tailwind)
# =============================================================================
FROM node:20-alpine

WORKDIR /app

# Copie des fichiers de dépendances pour maximiser le cache Docker
COPY package.json ./

# Installation des dépendances Node.js
RUN npm install

# Copie de l'ensemble de la configuration et du code source
COPY . .

EXPOSE 5173

# Lancement du serveur Vite exposé sur 0.0.0.0 pour l'accès depuis l'hôte / LXC
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
