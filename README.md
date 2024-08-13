## SEGMENTAÇÃO DE CLIENTES BASEADO NO USO DO CARTÃO DE CRÉDITO

O conjunto de dados para este dataset consiste no comportamento de uso do cartão de crédito dos clientes, com cerca de 9000 titulares de cartão de crédito ativos durante os últimos 6 meses com 18 características comportamentais. A segmentação dos clientes pode ser utilizada para definir estratégias de marketing.

**OBJETIVO**     
Identificar o perfil de clientes de cartão de crédito através de comportamentos relacionados as transações bancárias e extrair insights para área de marketing realizar estratégias e campanhas direcionadas para cada perfil de cliente.

**INFO DATASET:**
- CUST_ID: Identificação do titular do cartão de crédito (Categórico)
- BALANCE: Valor do saldo restante em sua conta para fazer compras
- BALANCE_FREQUENCY: Com que frequência o saldo é atualizado, pontuação entre 0 e 1 (1 = frequentemente atualizado, 0 = não atualizado com frequência)
- PURCHASES: Valor das compras feitas a partir da conta
- ONEOFF_PURCHASES: Maior valor de compra feito de uma só vez
- INSTALLMENTS_PURCHASES: Valor das compras feitas em parcelas
- CASH_ADVANCE: Adiantamento em dinheiro fornecido pelo usuário
- PURCHASES_FREQUENCY: Com que frequência as compras são feitas, pontuação entre 0 e 1 (1 = frequentemente compradas, 0 = não compradas com frequência)
- ONEOFFPURCHASESFREQUENCY: Com que frequência as compras são feitas de uma só vez (1 = frequentemente compradas, 0 = não compradas com frequência)
- PURCHASESINSTALLMENTSFREQUENCY: Com que frequência as compras parceladas são feitas (1 = frequentemente feitas, 0 = não feitas com frequência)
- CASHADVANCEFREQUENCY: Com que frequência o adiantamento em dinheiro é pago
- CASHADVANCETRX: Número de transações feitas com "Adiantamento em Dinheiro"
- PURCHASES_TRX: Número de transações de compra realizadas
- CREDIT_LIMIT: Limite de crédito do cartão para o usuário
- PAYMENTS: Valor do pagamento feito pelo usuário
- MINIMUM_PAYMENTS: Valor mínimo dos pagamentos feitos pelo usuário
- PRCFULLPAYMENT: Percentual do pagamento total pago pelo usuário
- TENURE: Tempo de serviço do cartão de crédito para o usuário
