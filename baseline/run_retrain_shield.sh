declare -a envs=("quadcopter" "pendulum_continuous" "cartpole" "selfdriving" "lanekeeping" "4-car-platoon" "8-car-platoon" "oscillator" "cartpole_change_l" "pendulum_change_m" "pendulum_change_l" "selfdriving_obs")

mkdir logs

for env in "${envs[@]}"
do
    python benchmarks/"$env".py --retrain_shield 2>&1 | tee logs/"$env"_retrain_shield.log
done