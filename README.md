# Quantitative Regime-Switching Engine & Derivative Hedging Terminal

## Architecture Institutionnelle
Application Streamlit de gestion quantitative de portefeuille multi-actifs intégrant :
- **Classification Walk-Forward de Régimes de Marché** : Purging et embargo stricts pour éliminer tout look-ahead bias, complété par une *Live Tail Inference* sur les dernières séances de marché.
- **Optimisation Robuste de Markowitz** : Rétrécissement linéaire de covariance (Ledoit-Wolf Shrinkage) avec contraintes institutionnelles de concentration (0 à 35% max).
- **Couverture Systématique Dérivée (Black-Scholes)** : Achat automatique d'un Put OTM à 5% (maturité 30 jours) lors du déclenchement du Régime 2 (Sell-Off) avec monitoring continu des grecques (Delta, Gamma, Thêta, Vega).
- **Backtesting Réaliste** : Intégration systématique de 15 bps de frictions de marché (10 bps de courtage + 5 bps de slippage) à chaque rotation.
- **Design System Futuriste** : Interface sombre type hedge fund institutionnel (Linear / Raycast / Bloomberg Terminal moderne) avec cartes Bento Grid en glassmorphism, typographies *Inter* et *JetBrains Mono*, et internationalisation bilingue FR/EN.

## Structure du Répertoire
```
quant_regime_engine/
│
├── .streamlit/
│   └── config.toml               # Palette sombre & configuration Streamlit
├── modules/
│   ├── __init__.py               # Package python
│   ├── translations.py          # Dictionnaire bilingue FR / EN
│   ├── data_loader.py            # Yfinance + buffer 400j + simulateur Markov 3-états
│   ├── feature_engineering.py    # Facteurs techniques et labellisation supervisée
│   ├── ml_regime_model.py        # Walk-forward avec purging, embargo & live tail inference
│   ├── portfolio_optimizer.py    # Covariance Ledoit-Wolf & allocation adaptative
│   ├── options_hedging.py        # Moteur Black-Scholes & calcul analytique des grecques
│   └── backtester.py             # Moteur de backtest avec déduction des frictions
│
├── app.py                        # Terminal Streamlit complet (5 onglets interactifs)
├── requirements.txt              # Dépendances de production
├── Dockerfile                    # Image conteneurisée Python 3.11-slim
├── docker-compose.yml            # Orchestration port 8501
└── README.md                     # Documentation technique
```

## Démarrage Rapide

### Option 1 : Localement via Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Option 2 : Déploiement Conteneurisé via Docker Compose
```bash
docker-compose up --build
```
L'interface est accessible à l'adresse `http://localhost:8501`.
