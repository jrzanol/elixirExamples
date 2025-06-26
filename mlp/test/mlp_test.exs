defmodule MLPTest do
  use ExUnit.Case
  doctest MLP

  test "greets the world" do
    assert MLP.hello() == :world
  end
end
