defmodule MLPDEF do
  defstruct input_size: 0, # Número total de entradas
            hidden_size: 0, # Número de neurônios na camada oculta.
            output_size: 0, # Número de neurônios na camada de saída.
            learning_rate: 0.01,  # Taxa de aprendizado.
            w1: [], # Pesos da camada de entrada para a camada oculta.
            w2: [], # Pesos da camada oculta para a camada de saída.
            b1: [], # Bias da camada oculta.
            b2: []  # Bias da camada de saída.

  # Definição do construtor para criar uma nova instância do MLPDEF.
  def new(input_size, hidden_size, output_size, learning_rate \\ 0.01) do
    %MLPDEF{
      input_size: input_size,
      hidden_size: hidden_size,
      output_size: output_size,
      learning_rate: learning_rate,
      w1: random_matrix(hidden_size, input_size),
      w2: random_matrix(output_size, hidden_size),
      b1: random_vector(hidden_size),
      b2: random_vector(output_size)
    }
  end

  def forward(%MLPDEF{w1: w1, w2: w2, b1: b1, b2: b2}, x) do
    z1 = add(np_dot(w1, x), b1)
    a1 = relu(z1)
    z2 = add(np_dot(w2, a1), b2)
    a2 = softmax(z2)
    %{a1: a1, a2: a2, z1: z1}
  end

  # Funções para a criação de matrizes e vetores aleatórios.
  defp random_matrix(rows, cols), do: (for _ <- 1..rows, do: (for _ <- 1..cols, do: :rand.uniform() - 0.5))
  defp random_vector(size), do: (for _ <- 1..size, do: :rand.uniform() - 0.5)

  # Funções de ativações e suas derivadas.
  defp relu(v), do: Enum.map(v, &max(0.0, &1))
  defp relu_derivative(v), do: Enum.map(v, &if(&1 > 0.0, do: 1.0, else: 0.0))
  defp softmax(xs) do
    exps = Enum.map(xs, &:math.exp/1)
    sum = Enum.sum(exps)
    Enum.map(exps, &(&1 / sum))
  end

  defp add(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x + y end)
  defp sub(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x - y end)
  defp dot(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()

  # Operações com matrizes e vetores.
  defp np_dot(m, v), do: Enum.map(m, fn row -> dot(row, v) end)
end

defmodule MLP do
  def read_dataset(path) do
    path |> Path.expand(__DIR__) |> File.stream! |> CSV.decode(headers: true) |> Enum.take(2)
  end

  def hello do
    IO.inspect(read_dataset("../../train.csv"))

    :world
  end
end
