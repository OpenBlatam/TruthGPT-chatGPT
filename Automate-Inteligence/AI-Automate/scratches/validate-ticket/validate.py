# coding: utf-8
########################################################################################################################
###########################################                       ######################################################
###########################################  READ ONLY LIBRARIES  ######################################################
###########################################                       ######################################################
########################################################################################################################
import json, logging, os, sys
sys.path.append(os.path.dirname(__file__))
log = logging.getLogger(__name__)
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators import MATInstanceInitOperator, MATInstanceExitOperator, MATPythonOperator, DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowException
from mat import *
from mat_error_management import MATWorkflowErrorManagement
from mat_success_management import MATWorkflowSuccessManagement
from mat_runtime import *
########################################################################################################################
#######################################                                  ###############################################
#######################################  CUSTOM LIBRARIES AND VARIABLES  ###############################################
#######################################                                  ###############################################
########################################################################################################################

try:
    import usecase
except:
    pass
import xmltodict

########################################################################################################################
##############################################                  ########################################################
##############################################  DAG DEFINITION  ########################################################
##############################################                  ########################################################
########################################################################################################################


default_args = {
    'owner': 'Iquall',
    'depends_on_past': False,
    'provide_context': True
}

dag = DAG(dag_id='7wl-9r5-czg', description='', start_date=datetime(2021,12,1), schedule_interval=None, catchup=False, on_failure_callback=MATWorkflowErrorManagement, on_success_callback=MATWorkflowSuccessManagement, default_args=default_args)


########################################################################################################################
##############################################                    ######################################################
##############################################  CUSTOM FUNCTIONS  ######################################################
##############################################                    ######################################################
########################################################################################################################

def check_ip_duplicate(ip,user,passwd):
    url = "https://10.150.57.28/arsys/services/ARService?server=arbal&webService=WS_ConsultaNE"
    body='''<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:WS_AltaElementos_WO">
            <soapenv:Header>
                <urn:AuthenticationInfo>
                    <urn:userName>{}</urn:userName>
                    <urn:password>{}</urn:password>
                    <!--Optional:-->
                    <urn:authentication>?</urn:authentication>
                    <!--Optional:-->
                    <urn:locale>?</urn:locale>
                    <!--Optional:-->
                    <urn:timeZone>?</urn:timeZone>
                </urn:AuthenticationInfo>
            </soapenv:Header>
            <soapenv:Body>
                <urn:Consulta_IP>
                    <urn:IP>{}</urn:IP>
                    <urn:startRecord>?</urn:startRecord>
                    <urn:maxLimit>?</urn:maxLimit>
                </urn:Consulta_IP>
            </soapenv:Body>
        </soapenv:Envelope>'''.format(user,passwd,ip)
    headers = {'content-type': 'text/xml','SOAPAction':''}
    response = requests.post(url,data=body,headers=headers,verify=False)
    xml = xmltodict.parse(response.content)
    if "soapenv:Fault" in xml["soapenv:Envelope"]["soapenv:Body"]:
        return False
    else:
        results_remedy = xml["soapenv:Envelope"]["soapenv:Body"]["ns0:Consulta_IPResponse"]["ns0:getListValues"]
        if not isinstance(results_remedy, dict):
            return True
        else:
            return False


def verificacion_ci(**kwargs):
    job = kwargs.get("job")
    inv = Inventory()
    job.log.info("Verificacion CI")
    module = "8qg-e46-lmw"
    formdata = getForm(kwargs)
    batch_id = formdata["data"]["batchId"]
    res_data = inv.get(module=module,query='data.state=In Progress&data.batchId={}'.format(batch_id))

    # Realizar la consulta de requerimientos
    for data in res_data:

        ## inicializo variables de influx para generaciÃ³n de estadÃ­sticas
        fields_influx = {"responsables_incorrectos": 0, "elemento_no_encontrado": 0, "elemento_duplicado": 0, "campos_incompletos": 0}
        tags_influx = { "total_errores": 0}


        qualification = "REQUERIMIENTO DE ALTA DE ELEMENTOS"
        credential = "Remedy_Prod_IPBH"
        cpasswd_user = inv.get(module='matcredentials',field='password',query='data.credentialName={}'.format(credential))[0]['password']
        cipher = crypto.aes256.MATCipher()
        passwd=cipher.decrypt(cpasswd_user)
        #passwd = "Temporal2021"
        user = inv.get(module='matcredentials',field='username',query='data.credentialName={}'.format(credential))[0]['username']
        url = "https://10.150.57.28/arsys/services/ARService?server=arbal&webService=WS_ConsultaNE"
        #name_ci = "MXMOREMZ0244ONT01"
        name_ci = data["data"]["networkElement"]
        ip = data["data"]["ipAddress"]
        #responsable_2 = data["data"]["responsable_2"]
        #responsable_3 = data["data"]["responsable_3"]
        tabla_escalacion = data["data"]["tabla_escalacion"]
        headers = {'content-type': 'text/xml','SOAPAction':''}
        module = "8qg-e46-lmw"
        body='''<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:WS_AltaElementos_WO">
            <soapenv:Header>
                <urn:AuthenticationInfo>
                    <urn:userName>{}</urn:userName>
                    <urn:password>{}</urn:password>
                    <!--Optional:-->
                    <urn:authentication>?</urn:authentication>
                    <!--Optional:-->
                    <urn:locale>?</urn:locale>
                    <!--Optional:-->
                    <urn:timeZone>?</urn:timeZone>
                </urn:AuthenticationInfo>
            </soapenv:Header>
            <soapenv:Body>
                <urn:Consulta_Nombre>
                    <urn:Nombre_CI>{}</urn:Nombre_CI>
                    <urn:startRecord>?</urn:startRecord>
                    <urn:maxLimit>?</urn:maxLimit>
                </urn:Consulta_Nombre>
            </soapenv:Body>
        </soapenv:Envelope>'''.format(user,passwd,name_ci)
        try:
            response = requests.post(url,data=body,headers=headers,verify=False)
            job.log.info("Remedy API Request, user: {},url :{}".format(user,url))
            job.log.info("Remedy Response:{}".format(response.content))
            xml = xmltodict.parse(response.content)
            #job.log.info(xml)
            job.log.info(json.dumps(xml, indent=4))
        except:
            job.log.error("No se pudo establecer conexiÃ³n con Remedy")
            for data in res_data:
                ipAddress = data['data']['ipAddress']
                state_remedy = "NOK"
                obs_remedy = "Error de conectividad"
                data_nueva = {"remedy":state_remedy,'obs_remedy': obs_remedy}
                inv.update(module=module, data = data_nueva, query='data.ipAddress={}&data.state=In Progress'.format(data["data"]["ipAddress"]))
            job.final_status(30001, "Error de conectividad")
            raise AirflowException("Error de conectividad")
        if "soapenv:Fault" in xml["soapenv:Envelope"]["soapenv:Body"]:
            mat_add_row_report(report="Reporte de Remedy", section="Gestor Remedy", fields={
                "Equipo": {
                    "type": "string",
                    "value": data["data"]["networkElement"]
                },
                "IP": {
                    "type": "string",
                    "value": data["data"]["ipAddress"]
                },
                "ID del CI": {
                    "type": "string",
                    "value": "Error"
                },
                "Model": {
                    "type": "string",
                    "value": "Error"
                },
                "Vendor": {
                    "type": "string",
                    "value": "Error"
                },
                "Gestor nativo": {
                    "type": "string",
                    "value": "Error"
                },
                "Subcategoria" : {
                    "type": "string",
                    "value": "Error"
                },
                "Ubicacion" : {
                    "type": "string",
                    "value": "Error"
                },
                "Responsables" : {
                    "type": "string",
                    "value": "Error"
                }
            })
            data["data"]["remedy"] = "NOK"
            data["data"]["obs_remedy"] = "Elemento no encontrado"
            fields_influx["elemento_no_encontrado"] = 1
            tags_influx["total_errores"] = tags_influx["total_errores"] + 1
            data_nueva = {"remedy":data["data"]['remedy'],'obs_remedy': data["data"]['obs_remedy']}
            inv.update(module=module, data = data_nueva, query='data.ipAddress={}&data.state=In Progress'.format(data["data"]["ipAddress"]))
        else:
            try:
                data["data"]["obs_remedy"] = ""
                ###### verifico que todos los campos vengan completos ####
                results_remedy = xml["soapenv:Envelope"]["soapenv:Body"]["ns0:Consulta_NombreResponse"]["ns0:getListValues"]
                if not isinstance(results_remedy, dict):
                    for item in results_remedy:
                        if ip == item["ns0:IP"]:
                            results_remedy = item
                            data["data"]["remedy"] = "NOK"
                            data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "Hostname duplicado. "
                            fields_influx["elemento_duplicado"] = 1
                            tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                            break
                ip_duplicada = check_ip_duplicate(ip,user,passwd)
                if ip_duplicada:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "IP duplicada. "
                    fields_influx["elemento_duplicado"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Responsable1" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 1 no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Responsable1"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 1 no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Responsable2" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 2 no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Responsable2"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 2 no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                #if "ns0:Responsable3" not in results_remedy:
                    #data["data"]["remedy"] = "NOK"
                    #data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 3 no estÃ¡ completo. "
                    #fields_influx["campos_incompletos"] = 1
                    #tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                #elif not results_remedy["ns0:Responsable3"]:
                    #data["data"]["remedy"] = "NOK"
                    #data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Responsable 3 no estÃ¡ completo. "
                    #fields_influx["campos_incompletos"] = 1
                    #tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:ID_CI" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de ID_CI no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:ID_CI"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de ID_CI no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Categoria" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de CategorÃ­a no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Categoria"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de CategorÃ­a no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Nombre_CI" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo Nombre_CI no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Nombre_CI"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Nombre_CI no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:IP" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de IP no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:IP"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de IP no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Subcategoria" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de SubcategorÃ­a no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Subcategoria"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de SubcategorÃ­a no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Gestor_Nativo" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Gestor Nativo no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Gestor_Nativo"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Gestor Nativo no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Fabricante" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Fabricante no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Fabricante"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Fabricante no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:ID_Ubicacion" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de ID_Ubicacion no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:ID_Ubicacion"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de ID_Ubicacion no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Nombre_Ubicacion" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Nombre Ubicacion no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Nombre_Ubicacion"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Nombre Ubicacion no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                if "ns0:Modelo_Version" not in results_remedy:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Modelo Version no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                elif not results_remedy["ns0:Modelo_Version"]:
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "El campo de Modelo Version no estÃ¡ completo. "
                    fields_influx["campos_incompletos"] = 1
                    tags_influx["total_errores"] = tags_influx["total_errores"] + 1

                responsable1 = results_remedy["ns0:Responsable1"] if results_remedy["ns0:Responsable1"] else ""
                responsable2 = results_remedy["ns0:Responsable2"] if results_remedy["ns0:Responsable2"] else ""
                #responsable3 = results_remedy["ns0:Responsable3"] if results_remedy["ns0:Responsable3"] else ""
                mat_add_row_report(report="Reporte de Remedy", section="Gestor Remedy", fields={
                    "Equipo": {
                        "type": "string",
                        "value": data["data"]["networkElement"]
                    },
                    "IP": {
                        "type": "string",
                        "value": data["data"]["ipAddress"]
                    },
                    "ID del CI": {
                        "type": "string",
                        "value": results_remedy["ns0:ID_CI"] if results_remedy["ns0:ID_CI"] else ""
                    },
                    "Model": {
                        "type": "string",
                        "value": results_remedy["ns0:Modelo_Version"] if results_remedy["ns0:Modelo_Version"] else ""
                    },
                    "Vendor": {
                        "type": "string",
                        "value": results_remedy["ns0:Fabricante"] if results_remedy["ns0:Fabricante"] else ""
                    },
                    "Gestor nativo": {
                        "type": "string",
                        "value": results_remedy["ns0:Gestor_Nativo"] if results_remedy["ns0:Gestor_Nativo"] else ""
                    },
                    "Subcategoria" : {
                        "type": "string",
                        "value": results_remedy["ns0:Subcategoria"] if results_remedy["ns0:Subcategoria"] else ""
                    },
                    "Ubicacion" : {
                        "type": "string",
                        "value": results_remedy["ns0:Nombre_Ubicacion"] + "/" + results_remedy["ns0:ID_Ubicacion"] if (results_remedy["ns0:Nombre_Ubicacion"] and results_remedy["ns0:ID_Ubicacion"]) else ""
                    },
                    "Responsables" : {
                        "type": "string",
                        "value": responsable1 + "/" + responsable2 #+ "/" + responsable3
                    }
                })

                #### verifico los responsables
                escalacion_ok = False
                if tabla_escalacion == "ACCESS":
                    responsable_1 = "BO IP BACKHAUL"
                    responsable_2 = "SUPP ACCESS"
                    #responsable_3 = "JORGE VILLEDA"
                    escalacion_ok = True
                elif tabla_escalacion == "CORE":
                    responsable_1 = "BO IP BACKHAUL"
                    responsable_2 = "SUPP IP CORE & DATA"
                    #responsable_3 = "CHRISTIAN VILLA"
                    escalacion_ok = True
                else:
                    job.log.warning("El valor ingresado en el campo de Tabla EscalaciÃ³n no es vÃ¡lido.")
                    data["data"]["remedy"] = "NOK"
                    data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "No se pudieron verificar los responsables. "
                if escalacion_ok:
                    if (responsable1 != responsable_1) or (responsable2 != responsable_2): # or (responsable3 != responsable_3):
                        data["data"]["remedy"] = "NOK"
                        data["data"]["obs_remedy"] = data["data"]["obs_remedy"] + "Los responsables son incorrectos"
                        fields_influx["responsables_incorrectos"] = 1
                        tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                    elif data["data"]["remedy"] != "NOK":
                        data["data"]["remedy"] = "OK"
                        data["data"]["obs_remedy"] = "-"
                data_nueva = {"remedy":data["data"]['remedy'],'obs_remedy': data["data"]['obs_remedy']}
                inv.update(module=module, data = data_nueva, query='data.ipAddress={}&data.state=In Progress'.format(data["data"]["ipAddress"]))

            except Exception as e:
                job.log.error(e)
                mat_add_row_report(report="Reporte de Remedy", section="Gestor Remedy", fields={
                    "Equipo": {
                        "type": "string",
                        "value": data["data"]["networkElement"]
                    },
                    "IP": {
                        "type": "string",
                        "value": data["data"]["ipAddress"]
                    },
                    "ID del CI": {
                        "type": "string",
                        "value": "Error"
                    },
                    "Model": {
                        "type": "string",
                        "value": "Error"
                    },
                    "Vendor": {
                        "type": "string",
                        "value": "Error"
                    },
                    "Gestor nativo": {
                        "type": "string",
                        "value": "Error"
                    },
                    "Subcategoria" : {
                        "type": "string",
                        "value": "Error"
                    },
                    "Ubicacion" : {
                        "type": "string",
                        "value": "Error"
                    },
                    "Responsables" : {
                        "type": "string",
                        "value": "Error"
                    }
                })
                data["data"]["remedy"] = "NOK"
                data["data"]["obs_remedy"] = "Elemento no encontrado"
                fields_influx["elemento_no_encontrado"] = 1
                tags_influx["total_errores"] = tags_influx["total_errores"] + 1
                data_nueva = {"remedy":data["data"]['remedy'],'obs_remedy': data["data"]['obs_remedy']}
                inv.update(module=module, data = data_nueva, query='data.ipAddress={}&data.state=In Progress'.format(data["data"]["ipAddress"]))
        ## InserciÃ³n de mÃ©tricas en influx
        job.log.info("Insertando estadÃ­sticas en influx...")
        status = mat_write_influx(measurement='remedy', tags= tags_influx, fields=fields_influx)
        if status==0:
            job.log.info("InserciÃ³n exitosa")
        else:
            job.log.error("Error de inserciÃ³n")
    job.final_status(11000, "OK")
    # Call next NW
    #mat = MATClient(host=None)
    #NW_ID = "bac-un4-bxi"
    #mat.networkWorkflow.jobs.run(NW_ID)

def verificacion_express(**kwargs):
    job = kwargs.get("job")
    inv = Inventory()
    qualification = "REQUERIMIENTO DE ALTA DE ELEMENTOS"
    credential = "Remedy_Prod_IPBH"
    cpasswd_user = inv.get(module='matcredentials',field='password',query='data.credentialName={}'.format(credential))[0]['password']
    cipher = crypto.aes256.MATCipher()
    passwd=cipher.decrypt(cpasswd_user)
    #passwd = "Temporal2021"
    user = inv.get(module='matcredentials',field='username',query='data.credentialName={}'.format(credential))[0]['username']
    #job.log.info(user)
    #job.log.info(passwd)
    url = "https://10.150.57.28/arsys/services/ARService?server=arbal&webService=WS_ConsultaNE"
    #name_ci = "MXMOREMZ0244ONT01"
    #name_ci = "10.190.0.31"
    #name_ci = "MXCABM01RTCORE02"
    name_ci = "TOBARITO"
    headers = {'content-type': 'text/xml','SOAPAction':''}
    module = "8qg-e46-lmw"
    body='''<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:WS_ConsultaNE">
        <soapenv:Header>
            <urn:AuthenticationInfo>
                <urn:userName>{}</urn:userName>
                <urn:password>{}</urn:password>
                <!--Optional:-->
                <urn:authentication>?</urn:authentication>
                <!--Optional:-->
                <urn:locale>?</urn:locale>
                <!--Optional:-->
                <urn:timeZone>?</urn:timeZone>
            </urn:AuthenticationInfo>
        </soapenv:Header>
        <soapenv:Body>
            <urn:Consulta_Nombre>
                <urn:Nombre_CI>{}</urn:Nombre_CI>
                <urn:startRecord>?</urn:startRecord>
                <urn:maxLimit>?</urn:maxLimit>
            </urn:Consulta_Nombre>
        </soapenv:Body>
    </soapenv:Envelope>'''.format(user,passwd,name_ci)
    response = requests.post(url,data=body,headers=headers,verify=False)
    job.log.info("Remedy API Request, user: {},url :{}".format(user,url))
    job.log.info("Remedy Response:{}".format(response.content))
    xml = xmltodict.parse(response.content)
    #job.log.info(xml)
    string = isinstance(xml, str)
    if string:
        job.log.info("Es un string")
    else:
        job.log.info("No es un string")
    job.log.info(json.dumps(xml, indent=4))


########################################################################################################################
###############################################                     ####################################################
###############################################  TASKS DEFINITIONS  ####################################################
###############################################                     ####################################################
########################################################################################################################


init = MATInstanceInitOperator(task_id='MAT_Initialize', dag=dag)
verificacion_ci = MATPythonOperator(task_id='verificacion_ci', python_callable=verificacion_ci, dag=dag)
#verificacion_express = MATPythonOperator(task_id='verificacion_express', python_callable=verificacion_express, dag=dag)
#resultado = MATPythonOperator(task_id='resultado', python_callable=resultado, dag=dag)
end = MATInstanceExitOperator(task_id='MAT_Finalize', dag=dag)


########################################################################################################################
################################################                  ######################################################
################################################  TASKS WORKFLOW  ######################################################
################################################                  ######################################################
########################################################################################################################

# PRUEBAS
#init >> verificacion_express >> end

# FLUJO FINAL
init >> verificacion_ci >> end