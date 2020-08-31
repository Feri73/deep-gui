lockfile=~/.clone_$1.lock
src=~/.android/avd/$1
dst=~/.android/avd/$2
tmp=~/.android/avd/tmptmp_$1
while [ ! -d $dst.avd ]; do
	echo locking $1 \($1 --\> $2\)
	while ! mkdir $lockfile; do sleep 10; done
	echo locked $1 \($1 --\> $2\)
	touch $lockfile/$2
	echo prechecking \($1 --\> $2\)
	if [ -d $tmp.avd ]; then
        	rm -rf $tmp.avd
	        rm $tmp.ini    
	fi
	echo copying $1.avd \($1 --\> $2\)
	cp -r $src.avd $tmp.avd
	echo copying $1.ini \($1 --\> $2\)
	cp $src.ini $tmp.ini
	echo moving $1 to $2 \($1 --\> $2\)
	~/android-sdk/tools/bin/avdmanager move avd -n $1 -r $2 || true
	if [ -d $src.avd ]; then
		echo something went wrong \($1 --\> $2\)
		rm -rf $tmp.avd
		rm $tmp.ini
		rm -rf $lockfile
		continue
	fi
	echo moving tmp.avd to $1.avd \($1 --\> $2\)
	mv $tmp.avd $src.avd
	echo moving tmp.ini to $1.ini \($1 --\> $2\)
	mv $tmp.ini $src.ini
	echo releasing $1.lock \($1 --\> $2\)
	rm -rf $lockfile
done
