FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run build

EXPOSE 8501

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "8501"]
