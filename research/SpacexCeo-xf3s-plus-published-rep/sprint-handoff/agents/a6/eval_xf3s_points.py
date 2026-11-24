"""a6: counterfactual close/gain/hedge points on c_xf3s teacher-forced data (totals: xf3s head = s 0.5, c 1.3, g 1.4)."""
import sys, json
sys.argv = ['x', sys.argv[1]]
exec(open('/home/user/delta/agents/a6/eval_hazard.py').read().split("if __name__ == '__main__':")[0])
pts = [(0.5, 1.3 * f, 1.4, f'close x{f}') for f in [0.88, 0.9, 0.92, 0.95, 0.96, 0.97, 0.975, 1.0, 1.03]]
pts += [(0.5, 1.3, 1.4 * f, f'g x{f}') for f in [0.9, 0.95, 1.05, 1.1]]
pts += [(0.5, 1.3 * 0.92, 1.4 * f, f'close x0.92 g x{f}') for f in [0.95, 1.05]]
pts += [(0.5 * f, 1.3, 1.4, f's x{f}') for f in [1.4, 1.6, 1.8, 2.0]]
R = []
for s, c, g, lab in pts:
    m = metrics(s, c, g); m['label'] = lab; R.append(m)
    print(f"{lab:22s} s={s:.3f} c={c:.3f} g={g:.3f} pos0={m['close_p_pos0']:.4f} natural={m['close_at_natural_end']:.3f} early/1k={m['close_haz_early_per1k']:.3f} rescue/1k={m['close_haz_trunc_late_per1k']:.3f} hedge7/1k={m['hedge7_per1k']:.3f} hedgeO/1k={m['hedgeO_per1k']:.3f} H={m['entropy']:.3f}")
json.dump(R, open('/home/user/delta/agents/a6/hazard_xf3s_points.json', 'w'), indent=0)
