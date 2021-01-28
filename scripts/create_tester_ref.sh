adb=~/android-sdk/platform-tools/adb

~/android-sdk/tools/bin/avdmanager create avd -n tester_ref --force -k "system-images;android-10;google_apis;x86" --device "5.4in FWVGA" -c 200M
echo hw.ramSize=6144 >> ~/.android/avd/tester_ref.avd/config.ini
screen -S tmp_ref_run -d -m bash -c "~/android-sdk/emulator/emulator -no-window -avd tester_ref"
echo waiting for device
$adb wait-for-device
echo waiting for complete boot
while :; do
	res=`$adb shell getprop sys.boot_completed`
	if [ 1 = ${res:0:1} ]; then
		break
	fi
	sleep 2
done
$adb push busybox /data/local/busybox
$adb push monkey.jar /data/local/mpp.jar
echo installing busybox
$adb shell "su -c ""chmod 755 /data/local/busybox; mount -o remount,rw -t yaffs2 /dev/block/mtdblock4 /system; /data/local/busybox cp /data/local/busybox /system/xbin; /system/xbin/busybox --install /system/xbin/; cp /data/local/mpp.jar /system/framework/mpp.jar; rm /data/local/mpp.jar; echo \"base=/system\" > /system/bin/monkey; echo \"export CLASSPATH=\\\$base/framework/mpp.jar\" >> /system/bin/monkey; echo \"for a in \\\"\\\$*\\\"; do\" >> /system/bin/monkey; echo \"    echo \\\"  bash arg:\\\" \\\$a\" >> /system/bin/monkey; echo \"done\"  >> /system/bin/monkey; echo \"exec app_process \\\$base/bin com.android.commands.monkey.Monkey \\\$*\" >> /system/bin/monkey; rm /system/app/PinyinIME.*; rm /system/app/LatinIME.*; rm /system/app/OpenWnn.*; mount -o ro,remount -t yaffs2 /dev/block/mtdblock4 /system; sync; reboot -p"""

screen -S tmp_ref_run -p 0 -X stuff "^C"
