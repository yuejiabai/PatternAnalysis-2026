"""Standard-library-only synthetic tests; no ADNI, torch, or image decoding."""
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import prepare_cv_folds as cv


class FoldTests(unittest.TestCase):
    def test_source_loader_fingerprints_and_identity_exclusion_without_decoding(self):
        class Subset:
            def __init__(self, dataset, indices):
                self.dataset, self.indices = dataset, indices
            def __getitem__(self, index):
                raise AssertionError('Image decoding forbidden')
        source = types.SimpleNamespace(samples=[], class_to_idx={'AD': 0, 'NC': 1})
        mapping, training, validation = {}, [], []
        for p in range(349):
            label = 0 if p < 90 else 1
            is_val = p < 18 or 90 <= p < 142
            for s in range(2 if p < 286 else 1):
                scan = 'scan{}x{}'.format(p, s)
                mapping[scan] = 'synthetic' + str(p)
                for position in range(20):
                    index = len(source.samples)
                    source.samples.append(('/synthetic/train/{}/{}_{}.jpeg'.format(
                        'AD' if label == 0 else 'NC', scan, position), label))
                    (validation if is_val else training).append(index)
        mapping['excluded'] = 'excluded_patient'
        test = types.SimpleNamespace(samples=[('/synthetic/test/excluded_0.jpeg', 0)])
        datasets = (Subset(source, training), Subset(source, validation), test)
        fake_dataset = types.ModuleType('dataset')
        fake_dataset.load_datasets = lambda: datasets
        fake_dataset.load_patient_map = lambda: mapping
        fake_dataset.get_image_id = lambda path: Path(path).stem.split('_')[0]
        fake_dataset.DATA_ROOT = Path('/synthetic')
        fake_data = types.ModuleType('torch.utils.data')
        fake_data.Subset = Subset
        config = dict(model='convnext', smoke_test=False, class_to_idx=source.class_to_idx,
                      split_fingerprint={name: cv.split_fingerprint([source.samples[i] for i in indices])
                                         for name, indices in [('train', training), ('validation', validation)]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            modules = {'dataset': fake_dataset, 'torch': types.ModuleType('torch'),
                       'torch.utils': types.ModuleType('torch.utils'), 'torch.utils.data': fake_data}
            with mock.patch.dict('sys.modules', modules), mock.patch.object(Path, 'is_file', return_value=True):
                manifest = cv.load_source(path)
                self.assertEqual(len(manifest['samples']), 12700)
                mapping['excluded'] = mapping['scan0x0']
                with self.assertRaises(cv.FoldError):
                    cv.load_source(path)
                mapping['excluded'] = 'excluded_patient'
                config['split_fingerprint']['train'] = 'wrong'
                path.write_text(json.dumps(config))
                with self.assertRaises(cv.FoldError):
                    cv.load_source(path)

    def records(self):
        rows = []
        for label, count in ((0, 90), (1, 259)):
            for patient in range(count):
                identity = "synthetic_{}_{}".format(label, patient)
                for scan in range(1 + patient % 2):
                    scan_id = identity + "_scan" + str(scan)
                    for position in range(2):
                        rows.append(dict(patient=identity, scan=scan_id, label=label,
                                         path=scan_id + "_" + str(position) + ".jpeg"))
        return rows

    def test_partition_and_reproducibility(self):
        rows = self.records()
        manifest = cv.build_manifest(rows, {})
        self.assertEqual(manifest, cv.build_manifest(copy.deepcopy(rows), {}))
        self.assertEqual(manifest['patient_to_fold'],
                         cv.build_manifest(list(reversed(rows)), {})['patient_to_fold'])
        seen = set()
        for fold in manifest['folds']:
            val, train = set(fold['validation_patients']), set(fold['train_patients'])
            self.assertFalse(seen & val)
            self.assertFalse(train & val)
            seen.update(val)
            self.assertEqual(fold['validation_patient_class_counts']['0'], 18)
            self.assertEqual(fold['validation_patient_class_counts']['1'],
                             51 if fold['fold_id'] == 4 else 52)
            self.assertEqual({rows[i]['patient'] for i in fold['validation_indices']}, val)
            self.assertEqual({rows[i]['patient'] for i in fold['train_indices']}, train)
        self.assertEqual(len(seen), 349)
        self.assertEqual(sorted(i for f in manifest['folds'] for i in f['validation_indices']),
                         list(range(len(rows))))

    def test_reject_bad_sources(self):
        rows = self.records()
        for mutation in ('duplicate', 'mixed_patient', 'mixed_scan', 'missing_patient', 'bad_label'):
            bad = copy.deepcopy(rows)
            if mutation == 'duplicate':
                bad.append(copy.deepcopy(bad[0]))
            elif mutation == 'mixed_patient':
                bad[0]['scan'] = 'different_scan'
                bad[0]['label'] = 1
            elif mutation == 'mixed_scan':
                bad[0]['patient'] = 'different_patient'
            elif mutation == 'missing_patient':
                bad[0]['patient'] = ''
            else:
                bad[0]['label'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(cv.FoldError):
                cv.build_manifest(bad, {})
        with self.assertRaises(cv.FoldError):
            cv.build_manifest(rows, {}, expected=(349, 635, 12700, 90, 259))

    def test_fingerprints_detect_membership_labels_and_mapping(self):
        rows = self.records()
        original = cv.build_manifest(rows, {})['fingerprint']
        self.assertNotEqual(original, cv.build_manifest(rows[:-1], {})['fingerprint'])
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['patient'] == rows[0]['patient']:
                row['patient'] = 'renamed_synthetic_patient'
        self.assertNotEqual(original, cv.build_manifest(changed, {})['fingerprint'])
        self.assertNotEqual(cv.split_fingerprint([('a', 0)]), cv.split_fingerprint([('a', 1)]))

    def test_verify_detects_tampering_and_missing_artifacts(self):
        manifest = cv.build_manifest(self.records(), {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if os.name == 'posix':
                os.chmod(root, 0o700)
            for name, value in cv.artifacts(manifest).items():
                path = root / name
                path.write_text(json.dumps(value), encoding='utf-8')
                if os.name == 'posix':
                    os.chmod(path, 0o600)
            cv.write_or_verify(root, manifest, verify=True)
            path = root / 'fold_0.json'
            original = path.read_text()
            path.write_text('{}')
            with self.assertRaises(cv.FoldError):
                cv.write_or_verify(root, manifest, verify=True)
            path.write_text(original)
            path.unlink()
            with self.assertRaises(cv.FoldError):
                cv.write_or_verify(root, manifest, verify=True)

    @unittest.skipUnless(os.name == 'posix', 'Private output creation requires Rangpur/Linux')
    def test_private_output_and_no_overwrite(self):
        manifest = cv.build_manifest(self.records(), {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'private'
            cv.write_or_verify(output, manifest)
            cv.write_or_verify(output, manifest, verify=True)
            self.assertEqual((output / '.gitignore').read_text(), '*\n')
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for path in output.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(cv.FoldError):
                cv.write_or_verify(output, manifest)


if __name__ == '__main__':
    unittest.main()
