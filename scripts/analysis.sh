name=$1
logs_dir=$2
apps_dir=$3
options=$4
runs_per_app=$5
tools=`ls -d $logs_dir/*chunk* | sed -n 's/.*\/\(.*\)_tester.*/\1/p' | uniq`

# uncomment for android analysis
python ../src/analyze_logs.py --analysis simple --name $name  --runs-per-app $runs_per_app --runs-per-app-per-tester 1 --tags Coverage/Line Coverage/Block Coverage/Class Coverage/Method --tools $tools --apps-dir $apps_dir --logs-dir $logs_dir $options

# uncomment for web analysis
#python ../src/analyze_logs.py --analysis simple --name $name --runs-per-app $runs_per_app --runs-per-app-per-tester 1 --tags "Metrics/Activity Count" --tools $tools --apps-dir $apps_dir --logs-dir $logs_dir $options
