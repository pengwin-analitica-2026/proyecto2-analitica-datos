from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from week9_detection import (CBAM, DetectionModel, boxes_from_label,
                              decode_predictions, detection_loss,
                              detection_model_from_checkpoint,
                              labels_to_grid, match_detections, nms)


def logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-5, 1 - 1e-5))


class Week9DetectionTests(unittest.TestCase):
    def test_model_shape_and_cbam_position(self) -> None:
        model = DetectionModel(base_channels=8)
        self.assertIsInstance(model.backbone.cbam, CBAM)
        output = model(torch.zeros(2, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (2, 16, 16, 8))

    def test_ablacion_sin_cbam(self) -> None:
        """Rama de control de la ablacion: mismo modelo, sin atencion."""
        con = DetectionModel(base_channels=8, use_cbam=True)
        sin = DetectionModel(base_channels=8, use_cbam=False)

        self.assertIsInstance(con.backbone.cbam, CBAM)
        self.assertIsInstance(sin.backbone.cbam, torch.nn.Identity)

        # Misma forma de salida: las dos ramas son comparables.
        x = torch.zeros(1, 1, 256, 256)
        self.assertEqual(tuple(con(x).shape), tuple(sin(x).shape))

        # La unica diferencia de parametros son los de CBAM: la comparacion
        # aisla la atencion y no arrastra otros cambios de capacidad.
        n_cbam = sum(p.numel() for p in con.backbone.cbam.parameters())
        self.assertGreater(n_cbam, 0)
        self.assertEqual(sum(p.numel() for p in con.parameters())
                         - sum(p.numel() for p in sin.parameters()), n_cbam)

    def test_colision_de_celda_reubica_en_vez_de_perder(self) -> None:
        """Dos regiones con el centro en la misma celda deben sobrevivir ambas."""
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 20:28, 20:28] = 1    # sacro
        label[0, 21:27, 21:27] = 11   # coxal izquierdo, mismo centro de celda

        self.assertEqual(len(boxes_from_label(label)[0]), 2)

        stats: dict = {}
        target = labels_to_grid(label, stats=stats)

        # Las dos regiones quedan codificadas, y la reubicacion se reporta.
        self.assertEqual(int(target[..., 0].sum()), 2)
        self.assertEqual(stats["reubicadas"], 1)
        self.assertEqual(stats["descartadas"], 0)

        # Ambas clases estan presentes, no se duplico ninguna.
        clases = {int(fila[5:].argmax()) for fila in target[target[..., 0] == 1]}
        self.assertEqual(clases, {0, 1})

        # tx/ty siguen dentro del rango que sigmoid puede representar.
        activas = target[target[..., 0] == 1]
        self.assertTrue(bool(((activas[:, 1:3] >= 0) & (activas[:, 1:3] <= 1)).all()))

    def test_metrica_penaliza_falsos_positivos(self) -> None:
        """Las cajas que sobran deben contar como falsos positivos."""
        verdad = [{"class_id": 0, "box": [0.1, 0.1, 0.5, 0.5]}]
        acierto = {"class_id": 0, "box": [0.11, 0.11, 0.49, 0.49], "score": 0.9}
        sobrante = {"class_id": 0, "box": [0.7, 0.7, 0.9, 0.9], "score": 0.4}

        solo = match_detections([acierto], verdad)
        self.assertEqual((solo["tp"], solo["fp"], solo["fn"]), (1.0, 0.0, 0.0))

        # Misma deteccion buena, pero disparando una caja de mas.
        con_ruido = match_detections([acierto, sobrante], verdad)
        self.assertEqual((con_ruido["tp"], con_ruido["fp"]), (1.0, 1.0))
        # El IoU medio NO distingue los dos casos; por eso hace falta el resto.
        self.assertAlmostEqual(solo["iou_suma"], con_ruido["iou_suma"], places=6)

        # Region no detectada: cuenta como falso negativo, con IoU 0.
        vacio = match_detections([], verdad)
        self.assertEqual((vacio["tp"], vacio["fn"], vacio["iou_suma"]), (0.0, 1.0, 0.0))

    def test_target_encode_decode_round_trip(self) -> None:
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 8:24, 4:20] = 1
        label[0, 30:55, 38:61] = 21
        target = labels_to_grid(label)
        self.assertEqual(int(target[..., 0].sum()), 2)

        # Construye logits perfectos a partir del target para probar que la
        # decodificación invierte la codificación geométrica.
        pred = torch.full_like(target, -12.0)
        positives = target[..., 0] == 1
        pred[..., 0][positives] = 12.0
        pred[..., 1:5][positives] = logit(target[..., 1:5][positives])
        for index in torch.nonzero(positives):
            b, i, j = index.tolist()
            cls = int(target[b, i, j, 5:].argmax())
            pred[b, i, j, 5:] = -12.0
            pred[b, i, j, 5 + cls] = 12.0

        decoded = decode_predictions(pred, confidence=0.9)[0]
        truth = boxes_from_label(label)[0]
        self.assertEqual(len(decoded), len(truth))
        for gt in truth:
            candidate = next(item for item in decoded if item["class_id"] == gt["class_id"])
            self.assertTrue(torch.allclose(torch.tensor(candidate["box"]),
                                           torch.tensor(gt["box"]), atol=1e-4))

    def test_max_por_clase_descarta_duplicadas_que_el_nms_no_ve(self) -> None:
        """Dos cajas de la misma clase que casi no se solapan: el NMS deja las
        dos; con max_por_clase=1 queda solo la de mayor puntaje."""
        pred = torch.full((1, 16, 16, 8), -12.0)
        for (i, j), obj in (((3, 3), 3.0), ((12, 12), 0.5)):   # puntajes ~0.95 y ~0.62
            pred[0, i, j, 0] = obj
            pred[0, i, j, 1:3] = 0.0                            # centro de la celda
            pred[0, i, j, 3:5] = logit(torch.tensor([0.15, 0.15]))
            pred[0, i, j, 5] = 12.0                             # clase 0 (sacro)
        todas = decode_predictions(pred, confidence=0.3)[0]
        self.assertEqual(len(todas), 2)
        una = decode_predictions(pred, confidence=0.3, max_por_clase=1)[0]
        self.assertEqual(len(una), 1)
        self.assertAlmostEqual(una[0]["score"], max(p["score"] for p in todas), places=6)
        with self.assertRaises(ValueError):
            decode_predictions(pred, max_por_clase=0)

    def test_gamma_cero_equivale_a_sin_atencion(self) -> None:
        """Con gamma = 0 (su valor inicial) el modelo de tres etapas calcula lo
        mismo que el de control sin CBAM: la atención entra solo si gamma crece."""
        torch.manual_seed(0)
        con = DetectionModel(base_channels=8, cbam_stages=3, learnable_gamma=True).eval()
        sin = DetectionModel(base_channels=8, use_cbam=False).eval()
        sin.load_state_dict({k: v for k, v in con.state_dict().items()
                             if k in sin.state_dict()})
        x = torch.rand(1, 1, 64, 64)
        self.assertTrue(torch.allclose(con(x), sin(x), atol=1e-6))
        self.assertEqual(set(con.backbone.valores_gamma()), {"etapa_4", "etapa_8", "etapa_16"})
        # gamma recibe gradiente: se puede aprender.
        con.train(); con(x).sum().backward()
        self.assertIsNotNone(con.backbone.gammas["etapa_16"].grad)

    def test_checkpoint_viejo_carga_igual(self) -> None:
        """Los valores por defecto no cambian los nombres de parámetros."""
        model = DetectionModel(base_channels=8)
        ck = {"base_channels": 8, "use_cbam": True, "model": model.state_dict()}
        cargado = detection_model_from_checkpoint(ck)
        self.assertEqual(set(cargado.state_dict()), set(model.state_dict()))
        self.assertEqual(cargado.backbone.valores_gamma(), {})

    def test_nms_suppresses_overlapping_lower_score(self) -> None:
        boxes = torch.tensor([[0.1, 0.1, 0.6, 0.6],
                              [0.12, 0.12, 0.58, 0.58],
                              [0.7, 0.7, 0.9, 0.9]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(nms(boxes, scores, iou_threshold=0.5).tolist(), [0, 2])

    def test_loss_is_finite_and_backpropagates(self) -> None:
        model = DetectionModel(base_channels=4)
        image = torch.rand(1, 1, 64, 64)
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 18:46, 20:44] = 11
        pred = model(image)
        loss, metrics = detection_loss(pred, labels_to_grid(label))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["positive_cells"], 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()