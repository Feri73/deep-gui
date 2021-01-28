exp_dir=$1
apks_dir=$2
exc_apks_dir=$3
simplest=$4

rm .analysis_cache.pck
cache=0

add_analysis() {
	local name=$1
	local options=$2
	if (( cache == 1 )); then
		options="--use-cache $options"
	fi
	cache=1
	if [ ! -z "$exc_apks_dir" ]; then
		local exc_apks_d="--excluded-apps-dir $exc_apks_dir"
	fi
	./analysis.sh $name $exp_dir/tb_otest_logs $apks_dir "$exc_apks_d $options"
	mkdir $exp_dir/tmpdir
	mv $exp_dir/tb_otest_logs/*$name* $exp_dir/tmpdir/
	mv $exp_dir/tmpdir $exp_dir/tb_otest_logs/$name
}

#add_analysis nt_none-nr_none-na_none-sa_mean_app_run "--ignore-missing args --summary-action mean --summary-axes app run --kstest-ref monkey --kstest-alt less"
add_analysis nt_mean-nr_monkey-na_run-sa_mean_app_run "--ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --kstest-ref monkey --kstest-alt less"
if (( simplest == 1 )); then exit 0; fi
#add_analysis nt_none-nr_none-na_none-sa_mean_run "--ignore-missing args --summary-action mean --summary-axes run --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_ent_run_time "--ignore-missing args --summary-action entropy --summary-axes run time --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_app_run_w_ent_run_time_tool "--ignore-missing args --summary-action mean --summary-axes app run --summary-param weights-entropy-run-time-tool --kstest-ref monkey --kstest-alt less"


#------------------------------------------
#add_analysis nt_none-nr_none-na_none-sa_std_run_time "--use-cache --ignore-missing args --summary-action std --summary-axes run time --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_range_run_time "--use-cache --ignore-missing args --summary-action range --summary-axes run time --kstest-ref monkey --kstest-alt less"

#add_analysis nt_none-nr_none-na_none-sa_std_run_time_tool "--use-cache --ignore-missing args --summary-action std --summary-axes run time tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_app_run_w_std_run_time_tool "--use-cache --ignore-missing args --summary-action mean --summary-axes app run --summary-param weights-std-run-time-tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_range_run_time_tool "--use-cache --ignore-missing args --summary-action range --summary-axes run time tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_run_time "--use-cache --ignore-missing args --summary-action mean --summary-axes run time --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_app_run_w_range_run_tool "--use-cache --ignore-missing args --summary-action mean --summary-axes app run --summary-param weights-range-run-tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_app_run_w_range_run_time_tool "--use-cache --ignore-missing args --summary-action mean --summary-axes app run --summary-param weights-range-run-time-tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_none-nr_none-na_none-sa_mean_app_run_w_std_run_tool "--use-cache --ignore-missing args --summary-action mean --summary-axes app run --summary-param weights-std-run-tool --kstest-ref monkey --kstest-alt less"

#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run_time "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run time --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run_max_time "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action max --summary-axes time --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run_min_time "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action min --summary-axes time --kstest-ref monkey --kstest-alt greater"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run_max_time_mean_app "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action max --summary-axes time --summary-action mean --summary-axes app --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_run_min_time_mean_app "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action min --summary-axes time --summary-action mean --summary-axes app --kstest-ref monkey --kstest-alt greater"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_app_run_w_range_run_tool "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-range-run-tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_app_run_w_range_run_time_tool "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-range-run-time-tool --kstest-ref monkey --kstest-alt less"
#add_analysis nt_mean-nr_monkey-na_run-sa_mean_app_run_w_std_run_tool "--use-cache --ignore-missing args --norm-type mean --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-std-run-tool --kstest-ref monkey --kstest-alt less"


#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_app_run "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_run "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_run_time "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run time"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_run_max_time "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action max --summary-axes time"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_run_max_time_mean_app "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes run --summary-action max --summary-axes time --summary-action mean --summary-axes app"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_app_run_w_range_run_tool "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-range-run-tool"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_app_run_w_range_run_time_tool "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-range-run-time-tool"
#add_analysis nt_zscore-nr_monkey-na_run-sa_mean_app_run_w_std_run_tool "--use-cache --ignore-missing args --norm-type zscore --norm-ref monkey --norm-axes run --summary-action mean --summary-axes app run --summary-param weights-std-run-tool"

