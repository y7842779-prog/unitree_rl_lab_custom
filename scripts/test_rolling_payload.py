"""Check payload physics, history order, state pairing and censored metrics."""
import math
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from rolling_payload_support import (CargoMonitor, METRICS, build_scene, indices,
    initial_samples, metrics, observation, sha)

SCENE = Path('/home/hyc/unitree_mujoco/unitree_robots/go2_Oracle/scene.xml')


def rows():
    return [dict(time_before_s=i*.02,time_after_s=(i+1)*.02,scored=int(i>=100),
        roll=.2 if i>=100 else 1.,pitch=0.,gx=0.,gy=math.sin(.2),gz=-math.cos(.2),
        vx=.3,vy=.4,wz=0.,cmd_vx=.6,cmd_vy=0.,cmd_wz=0.,
        x_progress=(i+1)*.006,y_progress=0.,cargo_lost=0,terminated=0,timeout=0,
        ball_rel_x=.01,ball_rel_y=0.,ball_rel_z=.097,predicted_mass_kg='',true_mass_kg=4.)
        for i in range(600)]


class RollingChecks(unittest.TestCase):
    def test_compiled_mass_inertia_and_free_joint(self):
        before = sha(SCENE.parent/'go2.xml')
        original = mujoco.MjModel.from_xml_path(str(SCENE))
        with tempfile.TemporaryDirectory() as tmp:
            nominal = None
            for mass in (2.,4.,6.,8.):
                model,info = build_scene(SCENE,Path(tmp)/f'm{mass}',mass)
                idx = indices(model)
                nonball = np.arange(model.nbody)!=idx['ball']
                np.testing.assert_array_equal(model.body_mass[nonball],original.body_mass[nonball])
                np.testing.assert_array_equal(model.body_inertia[nonball],original.body_inertia[nonball])
                np.testing.assert_array_equal(model.geom_friction,original.geom_friction)
                if nominal is None:
                    nominal = model.body_mass[nonball].copy()
                np.testing.assert_array_equal(model.body_mass[nonball],nominal)
                np.testing.assert_allclose(model.body_inertia[idx['ball']],.4*mass*.04**2)
                self.assertEqual((model.nq,model.nv,model.nu),(26,24,12))
                self.assertAlmostEqual(info['total_mass_kg'],15.206408+mass)
                data = mujoco.MjData(model)
                data.qpos[idx['ball_qidx']+2] = 2.
                data.qvel[idx['ball_vidx']] = .5
                data.qvel[idx['ball_vidx']+4] = 1.
                old = data.qpos[idx['ball_qidx']:idx['ball_qidx']+7].copy()
                mujoco.mj_step(model,data)
                self.assertGreater(data.qpos[idx['ball_qidx']],old[0])
                self.assertFalse(np.array_equal(data.qpos[idx['ball_qidx']+3:idx['ball_qidx']+7],old[3:]))
        self.assertEqual(sha(SCENE.parent/'go2.xml'),before)

    def test_history_is_term_major(self):
        frames = [np.arange(45,dtype=np.float32)+i*100 for i in range(26)]
        history = []
        for frame in frames:
            obs,history = observation(frame,history,25)
        expected = np.concatenate([frames[-1]]+[np.stack(frames[-25:])[:,a:b].reshape(-1)
            for a,b in ((0,3),(3,6),(6,9),(9,21),(21,33),(33,45))])
        np.testing.assert_array_equal(obs,expected)
        self.assertEqual(len(obs),1170)
        history = []
        for frame in frames:
            short,history = observation(frame,history,5)
        self.assertEqual(len(short),270)
        expected = np.concatenate([frames[-1]]+[np.stack(frames[-5:])[:,a:b].reshape(-1)
            for a,b in ((0,3),(3,6),(6,9),(9,21),(21,33),(33,45))])
        np.testing.assert_array_equal(short,expected)

    def test_formulas_and_sample_window(self):
        result = metrics(rows())
        self.assertEqual(result['scored_samples'],500)
        self.assertAlmostEqual(result['roll_rms'],.2)
        self.assertAlmostEqual(result['flatness_penalty'],math.sin(.2)**2)
        self.assertAlmostEqual(result['linear_xy_vector_rmse'],.5)
        self.assertEqual(result['transport_completed'],1)

    def test_dropped_ball_never_improves_loaded_metric(self):
        raw = rows()
        for row in raw[200:]:
            row.update(cargo_lost=1,vx=.6,vy=0.,roll=0.)
        result = metrics(raw)
        self.assertEqual(result['scored_samples'],100)
        self.assertAlmostEqual(result['linear_xy_vector_rmse'],.5)
        self.assertLess(result['full_window_linear_xy_vector_rmse'],.5)
        self.assertEqual(result['transport_completed'],0)
        self.assertEqual(result['robot_completed'],1)
        self.assertEqual(result['cargo_lost'],1)

    def test_invalid_and_early_failure_retained(self):
        raw = rows()
        for r in raw:
            r['gz'] *= 9.81
        result = metrics(raw)
        self.assertGreater(result['invalid_samples'],0)
        self.assertTrue(all(result[k]=='' for k in METRICS))
        raw = rows()[:50]
        raw[-1]['terminated'] = 1
        result = metrics(raw)
        self.assertEqual(result['failed'],1)
        self.assertEqual(result['scored_samples'],0)

    def test_loss_monitor_and_new_seeds(self):
        monitor = CargoMonitor()
        values = dict(ball_rel_x=.21,ball_rel_y=0.,ball_floor_contacts=0)
        for i in range(4):
            self.assertEqual(monitor.update(values,(i+1)*.02),0)
        self.assertEqual(monitor.update(values,.1),1)
        self.assertEqual(monitor.update(dict(values,ball_rel_x=0.),.12),1)
        old = {s['evaluation_seed'] for seed in (20261025,20261026) for s in initial_samples(seed,10)}
        self.assertFalse(old & {s['evaluation_seed'] for s in initial_samples(2026102601,10)})


if __name__ == '__main__':
    unittest.main()
