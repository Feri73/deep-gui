cur_i=$1
coverage_dir=$2
screenshots_dir=$3
merged_path=$4
apk=$5
adb=$6

$adb emu screenrecord screenshot $screenshots_dir/$cur_i.png

coverage_path=$coverage_dir/$cur_i.ec
$adb shell am broadcast -a edu.gatech.m3.emma.COLLECT_COVERAGE
$adb pull /mnt/sdcard/coverage.ec $coverage_path
while [ ! -f $coverage_path ]; do sleep 1; done
./prnt.sh "coverage pulled for $apk; adb=$adb"
$adb shell rm /mnt/sdcard/coverage.ec
if [ -f $merged_path ]; then
        java -cp /home/$USER/deep-gui/scripts/emma.jar emma merge -in $coverage_path -in $merged_path -out $merged_path.tmp
        rm $merged_path
        mv $merged_path.tmp $merged_path
else
        cp $coverage_path $merged_path
fi
java -cp /home/$USER/deep-gui/scripts/emma.jar emma report -r txt --in $apk.em -in $merged_path -Dreport.txt.out.file=$coverage_path.txt
