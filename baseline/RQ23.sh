declare -a envs=("quadcopter" "pendulum_continuous" "cartpole")

mkdir logs

for num in {1..10}
do
    for env in "${envs[@]}"
    do
        python benchmarks/"$env".py --nn_test --shield_test --test_episodes=1000 2>&1 | tee logs/"$env"_"$num".log
    done
done