#!/bin/bash
name=$1
id=$2
apps_dir=$3
tool_name=$5
logs_name=$4-$tool_name
use_deep=$6
run_count=$7
base_dir=$8
port=$((5554 + 2 * $id))
adb="/home/ferisampad/android-sdk/platform-tools/adb -s emulator-$port"
emulator=/home/ferisampad/android-sdk/emulator/emulator
aapt=/home/ferisampad/android-sdk/build-tools/29.0.2/aapt
logcat_base_dir=$base_dir/logcat_$logs_name
coverage_base_dir=$base_dir/coverage_$logs_name
# actions_base_dir=$base_dir/actions_$log_name

# in seconds
install_wait=10
# in miliseconds
action_wait=300
#in seconds
max_time=3600
#max_time=900
#in seconds
max_boot_wait=60
#in seconds
max_kill_wait=60

on_emma_failed() {
	./prnt.sh "received emma fail in $name"
	emma_failed=1
	kill -9 $monkey_pid
}

trap on_emma_failed USR1

is_booted() {
	local res=`$adb shell getprop sys.boot_completed`
	if [ 1 = ${res:0:1} ]; then return 0; else return 1; fi
}

kill_emul(){
	$adb emu kill
	local st_time=$SECONDS
	while is_booted; do
		if (( SECONDS - st_time > max_kill_wait )); then
			./prnt.sh "force killing emulator $name"
			kill -9 $emulator_pid
			sleep 1
		fi
		./prnt.sh "waiting for $name to be killed"
		sleep 2
        done
}

start_emul(){
	should_restart=$1
	while :; do
		$emulator -no-window -no-audio -avd $name -no-snapshot-load -ports $port,$(($port + 1)) &
		emulator_pid=$!
		local st_time=$SECONDS
		local ct_time=$st_time
		while (( ct_time - st_time < max_boot_wait )); do
			if is_booted; then return 0; fi
			./prnt.sh "waiting for $name to boot"
			sleep 2
			ct_time=$SECONDS
		done
		kill -9 $emulator_pid
		sleep 1
		if (( should_restart )); then
			./prnt.sh "apparently there is a problem with $name. restarting emulator."
			recreate_emul
		else
			return 1
		fi
	done
}

recreate_emul(){
	rm -rf ~/.android/avd/$name.avd
	rm ~/.android/avd/$name.ini
	taskset -c 0 ~/clone_avd.sh tester_ref $name
}

run_app_exp() {
	emma_failed=0
	./prnt.sh "starting $apk in $name"
  	package=`$aapt d xmltree $apk AndroidManifest.xml | grep package | awk 'BEGIN {FS="\""}{print $2}'`

	while :; do
		recreate_emul
		./prnt.sh "starting emulator in $name"
		start_emul 1
		./prnt.sh "installing $apk in $name"
		$adb install -r $apk
		sleep $install_wait
		timeout 10s $adb shell monkey -p $package 1
		./prnt.sh "restarting emulator in $name"
		kill_emul
		if start_emul 0; then break; fi
	done

	logcat_dir=$logcat_base_dir/$(basename $apk)_$round/
	mkdir -p $logcat_dir
	./logcat.sh $name $port $logcat_dir &
	logcat_pid=$!
	./coverage.sh $name $apk $coverage_base_dir $$ &
	coverage_pid=$!

	./prnt.sh "running monkey in $name for $package of $apk"
	SECONDS=0
	tout=$max_time
	if (( use_deep == 1 )); then
		dport=$((3000 + id))
		monkey_args="--deep --port $dport"
		$adb forward tcp:$dport tcp:$dport
		
		ctrport=$((5000 + id))
		./prnt.sh "sending reset weight request"
		echo rw | nc localhost $ctrport
	else
		monkey_args=""
	fi

	# actions_dir=$actions_base_dir/$(basename $apk)/$round
	# mkdir -p $actions_dir
		
	$adb shell input keyevent 82
	#$adb shell ime disable com.android.inputmethod.pinyin/.PinyinIME
	#$adb shell ime disable jp.co.omronsoft.openwnn/.OpenWnnJAJP
	#$adb shell ime disable com.android.inputmethod.latin/.LatinIME
	$adb shell ime disable com.example.android.softkeyboard/.SoftKeyboard
	# $adb shell ime list -s

	while :; do
		seed=$(od -N 4 -t uL -An /dev/urandom | tr -d " ")
		./prnt.sh "seed=$seed in $name"
  		timeout ${tout}s $adb shell monkey -p $package -s $seed $monkey_args --throttle $action_wait --ignore-crashes --ignore-timeouts --ignore-security-exceptions -v 1000000 &
		monkey_pid=$!
		wait $monkey_pid
		if (( emma_failed == 1 )); then
			./prnt.sh "emma failed in $name. trying again."
			break
		elif (( SECONDS < max_time )); then
			tout=$((max_time - SECONDS))
			# maybe here i need to recreate the emulator and also signal fatal error (and its handled after starting the app)
			./prnt.sh "error happened in monkey in $name. $tout seconds still remaining"
		else
			break
		fi
	done

	kill -9 $coverage_pid
	kill_emul
	kill -9 $logcat_pid
}

rm -r $logcat_base_dir/*
rm -r $coverage_base_dir/*
# rm -r $actions_base_dir/*

for round in $(seq 1 $run_count); do
	./prnt.sh "starting round $round in $name"
	for apk in `python -c "import glob; [print(x) for x in glob.glob('$apps_dir/*.apk')]"`; do
		while :; do
			run_app_exp
			if (( emma_failed == 0 )); then break; fi
			./prnt.sh "going for $apk again"
		done
	done
done
