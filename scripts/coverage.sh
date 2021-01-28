name=$1
apk=$2
dir=$3
parent_pid=$4
base_apk=$(basename $apk)
port=`ps aux | grep "\-a[v]d $name" | sed -n 's/.*-ports \(.*\),.*/\1/p'`
adb="/home/$USER/android-sdk/platform-tools/adb -s emulator-$port"

freq=60
delta_i=1
#in seconds
update_timeout=5
emma_fail_thr=10

on_fatal(){
	./prnt.sh "received fatal error in $name"
	delta_i=0
}

on_fatal_handled(){
	./prnt.sh "rceived fatal error handle in $name"
	delta_i=$orig_delta_i
}

call_update_logs(){
	emma_fail_cnt=$(</dev/shm/${name}_emma_fail_cnt)
	local cur_i=$i
	timeout ${update_timeout}s bash -c "/home/$USER/deep-gui/scripts/update_logs.sh $cur_i \"$coverage_dir\" \"$screenshots_dir\" \"$merged_path\" \"$apk\" \"$adb\""
	exit_status=$?
        if [[ $exit_status -eq 124 ]]; then
                echo nan > $coverage_dir/$cur_i.ec.txt
		./prnt.sh "single emma=nan in $name"
		emma_fail_cnt=$((emma_fail_cnt + 1))
	else
		emma_fail_cnt=$((emma_fail_cnt - 1))
		if (( emma_fail_cnt < 0 )); then emma_fail_cnt=0; fi
        fi

	./prnt.sh "emma_fail_cnt=$emma_fail_cnt	in $name"
	if (( emma_fail_cnt >= emma_fail_thr )); then
		./prnt.sh "sending emma failed to $parent_pid ($name)"
		rm -r $coverage_dir
		kill -10 $parent_pid
	fi

	echo $emma_fail_cnt >/dev/shm/${name}_emma_fail_cnt
}

trap on_fatal USR1
trap on_fatal_handled USR2

if [ -z "$(ls -A $dir/$name/$base_apk/)" ]; then
	round=0
else
	round=`ls $dir/$name/$base_apk/ | sort | tail -n 1`
	round=$((round + 1))
fi

echo 0 >/dev/shm/${name}_emma_fail_cnt
i=0
orig_delta_i=$delta_i
coverage_dir=$dir/$name/$base_apk/$round
screenshots_dir=$coverage_dir/screenshots
mkdir -p $coverage_dir
mkdir -p $screenshots_dir
merged_path=$coverage_dir/coverage.es
rm $merged_path
rm $coverage_dir/*.ec
rm $coverage_dir/*.txt

while :; do
	already_waited=0
	while (( already_waited < freq )); do
		./prnt.sh "should wait another $((freq - already_waited)) in $name"
		if (( delta_i == 0 )); then
			sleep $freq &
			wait $!
		else
			SECONDS=0
			sleep $((freq - already_waited)) &
			wait $!
			already_waited=$((already_waited + SECONDS))
		fi
	done
	i=$(( i + freq ))

	#end_i=$((i+freq-already_waited))
	#while (( i < end_i )); do i=$((i+delta_i)); echo i=$i in $name; sleep $delta_sleep; done
	#i=$((i+already_waited))

	call_update_logs &
done
