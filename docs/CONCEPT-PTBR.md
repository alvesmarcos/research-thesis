# Visão Geral

Construir uma ferramenta integrada com [fairseq](https://github.com/pytorch/fairseq) para automatização de **treinamento**, **validação** e **teste** para modelos de tradução *(machine translation)*.

<img src="https://translate.google.cn/about/images/hero-forbusiness.jpg" />


## Linhas de Pesquisa

A construção da ferramenta baseia-se em 3 (três) principais linhas de pesquisa.

- Treinamento de modelos de tradução com vocabulários dinâmicos.
	> Avaliar técnicas de treinamento para modelos de tradução adaptando o vocabulário de acordo com a entrada de novos dados.

- Inicialização de pesos.
	> Avaliar técnicas de inicialização de pesos para modelos de tradução (levando em consideração treinamento contínuo).

- Aprendizado de Máquina Automatizado *(AutoML)*.
	> Treinamento, pré-processamento e avaliação de maneira automatizada para modelos de tradução.

## Contexto

Testes computacionais para avaliar o **desempenho** da ferramenta automatizada proposta.

|Métricas|Modelo Não-Automatizado (MNA)|
|----------------|-------------------------------|
|Acurácia        | XX                            |

> **Nota:** O valor **XX** corresponde ao modelo inicial treinado sem ser de maneira automatizada.

## Hipóteses 

De maneira geral, a hipótese definida para o problema é a de que existe uma combinação de fatores, proposta pela ferramenta automatizada, que melhora o modelo inicial.

* **Hipótese Nula** (H<sub>0</sub>) - Não existe nenhuma combinação de fatores, proposta de forma automatizada, que aprimore o valor da acurácia do modelo inicial não automatizado.

* **Hipótese Alternativa** (H<sub>1</sub>) - Existe alguma combinação de fatores, proposta de forma automatizada, que aprimore o valor da acurácia do modelo inicial não automatizado.

## Seleção de Variáveis

* **Variáveis Dependentes** - Base de Treinamento (DBT),  Inicialização de Pesos (WI) e Técnica de Adaptação de Vocabulário (AVT).

* **Variáveis Independentes** - Acurácia (ACC).

## Fluxo
