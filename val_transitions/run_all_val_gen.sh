

for e in "CartPole-v1" "Pendulum-v1" "Reacher-v4" "SimpleEnv"
do
  for frac in $(seq 0 0.25 1)
  do
    echo "Generating validation transitions for $e Environment with Random action portion of $frac:"
    python generate_val_transitions.py --env $e --random_fraction $frac
  done
done