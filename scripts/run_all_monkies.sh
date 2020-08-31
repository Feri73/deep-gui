experiment_name=$1
tool_name=$2
use_deep=$3
agent_count=$4
rounds_count=$5
apps_dir=$6
#tb_logs_dir=$6/logs
base_dir=$7

list_descendants(){
	local ppid=$1
	local children=$(ps -o pid= --ppid "$ppid")
	for pid in $children; do
		list_descendants "$pid"
	done
	echo $children
}

kill_descendants(){
	kill `list_descendants $$`
}

trap 'kill_descendants; exit 130' SIGINT

rm -r ~/.clone_*.lock

#rm -r $tb_logs_dir/*
#mkdir -p $tb_logs_dir

echo "python update_tb.py monkey/ $experiment_name $tb_logs_dir $apps_dir &"
#tb_pid=$!

echo "$@" > $base_dir/args_$experiment_name-$tool_name
if (( use_deep == 1 )); then
	cp deep-gui/configs.yaml $base_dir/configs_$experiment_name_$tool_name.yaml
	cd deep-gui && python main.py &
	sleep 300
fi

echo_base_dir=$base_dir/${tool_name}_echos
mkdir -p $echo_base_dir
for id in $(seq 0 $((agent_count-1))); do
	./run_monkey.sh tester$id $id $apps_dir $experiment_name $tool_name $use_deep $rounds_count $base_dir > $echo_base_dir/$id.txt 2>&1 &
#	monkey_pids[${id}]=$!
done

while :; do sleep 10; done

#for pid in ${monkey_pids[*]}; do wait $pid; done
#wait $tb_pid
