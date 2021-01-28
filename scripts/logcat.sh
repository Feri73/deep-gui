while :
do
	/home/$USER/android-sdk/platform-tools/adb -s emulator-$2 logcat -v long time *:E >> $3/$1.log
	sleep 1
done
