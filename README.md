# Lista 5 - Trabalho final - Instabilidade das Estruturas

Após baixar/clonar o projeto, instale as dependências com uma das alternativas:

- Usando pip (padrão do Python):
```bash
pip install .
```

- Usando uv:
```bash
uv sync
```

Para executar o programa é necessário passar o caminho do arquivo de entrada na chamada da linha de comando. Dessa forma, para executar o programa com o arquivo de entrada situado em `examples/ex01/ex01.ie`, digita-se na linha de comando:
```bash
python main.py "examples/ex01/ex01.ie"
```

Para cada execução, são gerados:
- Arquivo de extensão `.out` - dados da análise, do modelo e resultados.
- Arquivo com sufixo *_axial_results* e extensão `.html` - gráficos dos resultados axiais (deslocamentos e esforços).
- Arquivo(s) com sufixo *_buckling_mode_x* e extensão `.html` - Gráfico(s) do(s) x primeiro(s) modo(s) de flambagem. O número de modos a serem extraídos é definido no arquivo de entrada.