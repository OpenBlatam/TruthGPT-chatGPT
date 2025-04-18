SELECT * FROM Tickets
WHERE
    ('Fecha de notificación+' >= '20/06/2022 12:00:00 a.m.') AND
    ('Grupo de propietarios' = 'ATCHOLA') AND
    ('Tipo de asignación' = 'Iniciado') AND
    ('Estado de la incidencia' != 'Cancelado') AND
    ('Estado de la incidencia' != 'Cerrado') AND
    (
        ('Fecha de última modificación' >= '02/02/2018 07:00' AND 'Fecha de última modificación' <= '02/02/2018 19:00') OR
        ('Fecha de notificación' >= '02/02/2018 07:00' AND 'Fecha de notificación' <= '02/02/2018 19:00')
    )