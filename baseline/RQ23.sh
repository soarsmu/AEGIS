declare -a envs=("quadcopter" "pendulum_continuous" "cartpole" "selfdriving")

mkdir logs

# for num in {1..10}
# do
for env in "${envs[@]}"
do
    echo "Running $env"
    python benchmarks/"$env".py --nn_test --shield_test --test_episodes=1000 2>&1 | tee logs/"$env"_new.log
done
# done

# for num in {1..10}
# do
# for env in "${envs[@]}"
# do
#     python benchmarks/"$env".py --retrain_shield 2>&1 | tee logs/"$env"_retrain_shield_"$num".log
# done
# done