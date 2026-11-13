param([string]$Origen, [string]$Destino, [string]$Pdf)
# Formato de la memoria (enunciado P2, ap. 9.1): A4, Calibri 11, interlineado sencillo, justificado.
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Origen, $false, $false)
    $cm = 28.3465

    # Página A4 y márgenes de 2,5 cm
    $ps = $doc.PageSetup
    $ps.PageWidth = 21.0 * $cm; $ps.PageHeight = 29.7 * $cm
    $ps.TopMargin = 2.5 * $cm; $ps.BottomMargin = 2.5 * $cm
    $ps.LeftMargin = 2.5 * $cm; $ps.RightMargin = 2.5 * $cm

    # Todo en Calibri; el código en línea también (sin Consolas)
    foreach ($nombre in @('Normal', 'CodeChar', 'Code')) {
        try { $st = $doc.Styles.Item($nombre); $st.Font.Name = 'Calibri'; $st.Font.Size = 11 } catch {}
    }

    foreach ($p in $doc.Paragraphs) {
        $f = $p.Format
        $f.LineSpacingRule = 0          # sencillo
        $f.SpaceBefore = 0
        $f.SpaceAfter = 4
        $p.Range.Font.Name = 'Calibri'
        $nivel = $p.OutlineLevel        # 1-9 = título, 10 = texto
        $enTabla = $p.Range.Information(12)   # wdWithInTable
        if ($nivel -eq 1) { $p.Range.Font.Size = 16; $p.Range.Font.Bold = $true; $f.SpaceAfter = 6; $f.Alignment = 0 }
        elseif ($nivel -lt 10) { $p.Range.Font.Size = 12; $p.Range.Font.Bold = $true; $f.SpaceBefore = 8; $f.SpaceAfter = 3; $f.Alignment = 0; $f.KeepWithNext = $true }
        elseif ($enTabla) { $p.Range.Font.Size = 11; $f.SpaceAfter = 0; $f.Alignment = 0 }
        elseif ($p.Range.InlineShapes.Count -gt 0) { $f.Alignment = 1; $f.KeepWithNext = $true }
        else { $p.Range.Font.Size = 11; $f.Alignment = 3 }   # justificado
    }

    # Figuras a 14 cm de ancho, centradas
    foreach ($s in $doc.InlineShapes) {
        $s.LockAspectRatio = -1
        $s.Width = 14.0 * $cm
    }

    # Línea de autoría (la exportación dejaba un identificador interno)
    $byline = $doc.Paragraphs.Item(2).Range
    $byline.MoveEnd(1, -1) | Out-Null
    $byline.Text = 'Robótica Inteligente Aplicada · Práctica 2 · [Nombres de las autoras] · Noviembre de 2026'
    $byline.Font.Size = 11; $byline.Font.Bold = $false
    $byline.ParagraphFormat.Alignment = 0

    # Fórmula: el LaTeX exportado se sustituye por una ecuación de Word
    foreach ($p in $doc.Paragraphs) {
        if ($p.Range.Text -like 'Q = 100*') {
            $r = $p.Range
            $r.MoveEnd(1, -1) | Out-Null
            $r.Text = 'Q=100 (∑ banda(sep)·(0,25+alin(desv))·Δt)/(1,25·T)-P_(sin arrancar)'
            $p.Format.Shading.BackgroundPatternColor = -16777216
            $p.Range.InsertParagraphAfter()
            $p2 = $p.Next()
            $r2 = $p2.Range
            $r2.MoveEnd(1, -1) | Out-Null
            $r2.Text = 'alin(desv)=max(0, 1-|desv|/(60°))'
            $doc.OMaths.Add($r2) | Out-Null
            $doc.OMaths.Add($r) | Out-Null
            foreach ($m in $doc.OMaths) { $m.BuildUp() }
            $p.Format.Alignment = 1; $p2.Format.Alignment = 1
            break
        }
    }

    # Tablas: columnas ajustadas al contenido y luego al ancho de página (sin cortar palabras)
    foreach ($t in $doc.Tables) {
        $t.AutoFitBehavior(1)
        $t.AutoFitBehavior(2)
        $t.Rows.AllowBreakAcrossPages = $false
    }

    # Los anexos empiezan en página nueva
    foreach ($p in $doc.Paragraphs) {
        if ($p.Range.Text -like 'Anexo A*') { $p.Format.PageBreakBefore = $true; $anexo = $p; break }
    }

    $doc.Repaginate()
    $total = $doc.ComputeStatistics(2)
    $finCuerpo = $anexo.Previous().Range.Information(3)
    Write-Output "Paginas del cuerpo (hasta Conclusiones): $finCuerpo"
    Write-Output "Paginas totales con anexos: $total"

    $doc.SaveAs2($Destino, 16)
    $doc.SaveAs2($Pdf, 17)
    $doc.Close($false)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
