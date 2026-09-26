"""Regression tests for real-world layouts learned from DocILE tuning. All inputs are SYNTHETIC text/geometry written for the test;
no dataset content or labels are used. Run: python3 -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extraction as X  # noqa: E402


def page(text, conf=0.95, boxes=None):
    lines = [l for l in text.split("\n")]
    return [{"page": 1, "text": text, "line_conf": [conf] * len(lines), **({"line_boxes": boxes} if boxes else {})}]


def fx(text, source="OCR", **kw):
    return X.extract_fields(page(text, **kw), source)


def val(f, k):
    return f[k]["value"] if f[k]["found"] else ("AMBIG" if f[k]["ambiguous"] else None)


class LabelVariants(unittest.TestCase):
    def test_invoice_number_label_forms(self):
        for text, want in (("INVOICE\nInvoice Number 747742", "747742"), ("INVOICE\nInvoice #:223656", "223656"), ("INVOICE\nINVOICE (#1850038409)", "1850038409"),
                           ("INVOICE\nInvoice N*: 22013", "22013"), ("INVOICE\ninveics Ne. 2", "2"), ("INVOICE\nInvoice # 16—04—050", "16-04-050"),
                           ("INVOICE\nInv. No.: A-77/3", "A-77/3"), ("Statement\nINVOICE NO. 051855 February 1, 2000", "051855")):
            f, _ = fx(text)
            self.assertEqual(val(f, "invoice_number"), want, text)

    def test_an_id_can_never_be_a_date_amount_or_phone(self):
        f, _ = fx("INVOICE\nInvoice No: 09/14/2020")
        self.assertIsNone(val(f, "invoice_number"))
        f, _ = fx("INVOICE\nInvoice No: 1,234.50")
        self.assertIsNone(val(f, "invoice_number"))
        f, _ = fx("INVOICE\nInvoice No: (212) 555-0187")
        self.assertIsNone(val(f, "invoice_number"))
        self.assertEqual(val(fx("INVOICE\nInventory number 55123")[0], "invoice_number"), None)      # 'inventory' is not 'invoice'

    def test_date_label_variants_and_formats(self):
        for text, want in (("INVOICE\nDate: January 21, 2002", "2002-01-21"), ("INVOICE\nDare: January 21, 2002", "2002-01-21"), ("INVOICE\nInvoice Date 27/11/22", "2022-11-27"),
                           ("INVOICE\nInvoice Date: 2026-09-02", "2026-09-02"), ("INVOICE\nDate issued: 3 March 2021", "2021-03-03")):
            f, _ = fx(text)
            self.assertEqual(val(f, "invoice_date"), want, text)
        f, w = fx("INVOICE\nInvoice Date: 4/11/22")
        self.assertEqual(val(f, "invoice_date"), "4/11/22"); self.assertTrue(any("ambiguous format" in x for x in w))   # day/month order unknown: kept as written, flagged

    def test_start_end_ship_dates_are_not_the_invoice_date(self):
        f, _ = fx("ORDER\nStart Date 09/27/22\nEnd Date 10/03/22\nDate Entered 04/29/22\nShip Date 05/01/22")
        self.assertIsNone(val(f, "invoice_date"))
        f, _ = fx("INVOICE\nInvoice No: 1\nDue Date: 2026-10-02")
        self.assertIsNone(val(f, "invoice_date")); self.assertEqual(val(f, "due_date"), "2026-10-02")

    def test_unlabelled_standalone_date_only_as_a_fallback_and_not_when_it_competes(self):
        self.assertEqual(val(fx("INVOICE\nAcme Widgets Inc\nJuly 25, 2000\nDescription Hours")[0], "invoice_date"), "2000-07-25")
        f, _ = fx("INVOICE\nAcme Widgets Inc\nJuly 25, 2000\nMarch 3, 1999\nDescription")
        self.assertEqual(val(f, "invoice_date"), "AMBIG")                                                          # two competing dates: never guess
        self.assertIsNone(val(fx("INVOICE\nService period 07/01/00 - 07/31/00\n")[0], "invoice_date"))            # a range is not a document date

    def test_prose_mentioning_invoice_number_is_not_a_header(self):
        text = "INVOICE\nPlease refer to our invoice number or return this copy when remitting\nFEDERAL ID #25-1126415"
        self.assertIsNone(val(fx(text)[0], "invoice_number"))


class HeaderValueTables(unittest.TestCase):
    def test_values_under_headers_are_aligned_by_position_not_order(self):
        head = "INVOICE NUMBER | INVOICE DATE CLIENT PURCHASE ORDER TERMS"
        vals = "2002041 2/8/00 NET 10 DAYS"
        hb = [["INVOICE", 46, 142, .95], ["NUMBER", 151, 254, .95], ["|", 395, 402, .9], ["INVOICE", 415, 511, .95], ["DATE", 521, 580, .95],
              ["CLIENT", 915, 1000, .95], ["PURCHASE", 1008, 1137, .95], ["ORDER", 1146, 1230, .95], ["TERMS", 2100, 2170, .95]]
        vb = [["2002041", 118, 238, .95], ["2/8/00", 654, 746, .95], ["NET", 2180, 2238, .95], ["10", 2248, 2284, .95], ["DAYS", 2295, 2370, .95]]
        f, _ = X.extract_fields(page("INVOICE\n" + head + "\n" + vals, boxes=[[["INVOICE", 400, 700, .95]], hb, vb]), "OCR")
        self.assertEqual((val(f, "invoice_number"), val(f, "invoice_date")), ("2002041", "2/8/00"))
        self.assertEqual(f["invoice_number"]["evidence_text"], vals)

    def test_address_text_before_the_values_does_not_shift_alignment(self):
        hb = [["Invoice", 100, 200, .9], ["#", 205, 220, .9], ["Invoice", 700, 800, .9], ["Date", 805, 870, .9]]
        vb = [["Yakima,", 20, 90, .9], ["WA", 95, 130, .9], ["98908", 135, 200, .9], ["644823-1", 230, 400, .9], ["08/26/18", 720, 850, .9]]
        f, _ = X.extract_fields(page("INVOICE\nInvoice # Invoice Date\nYakima, WA 98908 644823-1 08/26/18", boxes=[[["INVOICE", 10, 90, .9]], hb, vb]), "OCR")
        self.assertEqual(val(f, "invoice_number"), "644823-1"); self.assertEqual(val(f, "invoice_date"), "2018-08-26")   # day 26 > 12: unambiguous, normalised

    def test_net_and_gross_columns_stay_ambiguous(self):
        hb = [["Agency", 500, 560, .9], ["Commission", 565, 650, .9], ["Net", 800, 840, .9], ["Gross", 1000, 1060, .9]]
        vb = [["$672.00", 500, 590, .9], ["$", 790, 800, .9], ["3,808.00", 805, 900, .9], ["$", 990, 1000, .9], ["4,480.00", 1005, 1100, .9]]
        f, w = X.extract_fields(page("ORDER\nAgency Commission Net Gross\n$672.00 $ 3,808.00 $ 4,480.00", boxes=[[["ORDER", 10, 90, .9]], hb, vb]), "OCR")
        self.assertTrue(f["amount"]["ambiguous"]); self.assertEqual(sorted(f["amount"]["candidates"]), ["3808.00", "4480.00"])
        self.assertTrue(f["currency"]["ambiguous"])


class AmountRules(unittest.TestCase):
    def test_label_forms_and_ocr_damage(self):
        for text, want in (("INVOICE\nJob Total 82.91 an", "82.91"), ("INVOICE\nAmt Due= 1050. 00", "1050.00"), ("INVOICE\nPlease pay $1,463.70", "1463.70"),
                           ("INVOICE\nBalance Due 0.00", "0.00"), ("INVOICE\nTotal Due: EUR 430.50", "430.50")):
            f, _ = fx(text)
            self.assertEqual(val(f, "amount"), want, text)

    def test_tax_vendor_and_subtotals_are_not_the_amount(self):
        f, _ = fx("INVOICE\nSubtotal 100.00\nSales Tax Total $0.00\nVENDOR TOTAL 24.44\nTotal Spots 39")
        self.assertIsNone(val(f, "amount"))
        f, _ = fx("INVOICE\nSales Tax Total $0.00\nTotal $13,000.00")
        self.assertEqual(val(f, "amount"), "13000.00")

    def test_net_due_outranks_total_due_and_conflicts_in_a_tier_stay_ambiguous(self):
        f, w = fx("INVOICE\nTotal Due: 475.00\nNet Due: 0.00")
        self.assertEqual(val(f, "amount"), "0.00"); self.assertTrue(any("differ from the amount due" in x for x in w))
        f, _ = fx("INVOICE\nTotal 100.00\nTotal 250.00")
        self.assertEqual(val(f, "amount"), "AMBIG")

    def test_junk_after_a_dollar_sign_is_not_money(self):
        f, _ = fx("INVOICE\nTotal 1 day @ $1S00/day")
        self.assertIsNone(val(f, "amount"))


class DocumentTypeFailClosed(unittest.TestCase):
    def test_incidental_invoice_words_are_not_a_document_type(self):
        f, _ = fx("STATION: KRSJ-FM ORDER#: 3195902\nContact Invoices@MediaFinancial.com\nCONTRACT # FOR INVOICING 4423750\nPlease sign and return with invoice")
        self.assertNotEqual(f["document_type"]["value"], "invoice")

    def test_order_receipt_and_purchase_order_cues(self):
        for text, want in (("ORDER WORKSHEET\nREP HEADLINE# 9608657", "order"), ("SCHEDULING ORDER\nStation KXYZ", "order"), ("Thank you for your payment. The following was received", "receipt"),
                           ("PURCHASE ORDE\nNo. 9933", "purchase_order"), ("Sales Order\nCustomer PO 12", "order")):
            f, _ = fx(text)
            self.assertEqual(f["document_type"]["value"], want, text)

    def test_unknown_type_is_never_assumed_payable(self):
        f, _ = fx("Supplier: Acme Widgets Ltd\nTotal Due: EUR 5.00\nRef 991")
        self.assertFalse(f["document_type"]["found"])
        d = X.assess(f, "OCR")
        self.assertEqual(d["status_hint"], "REVIEW"); self.assertIn("document type could not be determined", d["reasons"]); self.assertFalse(d["payable_schema"])

    def test_order_types_route_to_review_never_to_payment(self):
        f, _ = fx("SCHEDULING ORDER\nSupplier: Acme Widgets Ltd\nInvoice No: 5\nTotal Due: EUR 5.00")
        self.assertEqual(X.assess(f, "OCR")["status_hint"], "REVIEW")


class SupplierRules(unittest.TestCase):
    def test_labelled_supplier_is_trusted_and_can_trigger_clarification(self):
        f, _ = fx("INVOICE\nSupplier: Imaginary Freight Co\nInvoice No: IFC-1\nTotal Due: 5.00")
        self.assertEqual(f["supplier_name"]["basis"], "label"); self.assertEqual(X.assess(f, "OCR")["status_hint"], "CLARIFY")

    def test_letterhead_supplier_never_triggers_a_clarification_email(self):
        f, _ = fx("Nordic Freight Services Inc\nINVOICE\nInvoice No: NF-1\nTotal Due: 5.00")
        self.assertEqual(val(f, "supplier_name"), "Nordic Freight Services Inc"); self.assertEqual(f["supplier_name"]["basis"], "letterhead")
        self.assertIn("letterhead", f["supplier_name"]["normalization_applied"])
        d = X.assess(f, "OCR")
        self.assertEqual(d["status_hint"], "REVIEW"); self.assertTrue(any("inferred from the letterhead" in r for r in d["reasons"]))

    def test_remit_to_block_is_a_labelled_source(self):
        f, _ = fx("INVOICE\nInvoice No: 7\nMake checks payable to: Harbor Supplies LLC\nTotal Due: EUR 9.00")
        self.assertEqual(val(f, "supplier_name"), "Harbor Supplies LLC")

    def test_headings_fragments_customers_and_addresses_are_rejected(self):
        for text in ("Order Printout\nINVOICE", "SCHEDULING ORDER\nINVOICE", "THE\nINVOICE", "MEMBER\nINVOICE", "from the program logs\nINVOICE", "Signed\nINVOICE",
                     "3636 Momentum Place\nINVOICE", "ARDSLEY, NEW YORK 10502\nINVOICE", "C/O THE NEW MEDIA FIRM\nINVOICE", "Estimate #\nINVOICE"):
            self.assertIsNone(val(fx(text)[0], "supplier_name"), text)
        f, _ = fx("INVOICE\nBILL TO:\nGlobal Tobacco Company Inc\n12 Main Street\nInvoice No: 5\nTotal Due: 5.00")
        self.assertNotEqual(val(f, "supplier_name"), "Global Tobacco Company Inc")                                  # the customer block is never the vendor

    def test_payment_sentence_is_not_a_remit_to_name(self):
        f, _ = fx("INVOICE\nSTATION. PAYMENT TO STATION WILL BE made within 30 days\nInvoice No: 5")
        self.assertIsNone(val(f, "supplier_name"))

    def test_station_call_sign_is_the_vendor_on_order_documents_but_never_on_invoices(self):
        f, _ = fx("SCHEDULING ORDER\nSTATION: KRSJ-FM ORDER#: 3195902 DATE: 04/22/2022")
        self.assertEqual(val(f, "supplier_name"), "KRSJ-FM")
        f, _ = fx("INVOICE\nInvoice No: 44\nSTATION: KRSJ-FM\nTotal Due: 5.00")
        self.assertNotEqual(val(f, "supplier_name"), "KRSJ-FM")                     # on a payable invoice the station is not trusted as the vendor
        hb = [["Advertiser", 100, 260, .9], ["Station", 700, 790, .9], ["Market", 1200, 1290, .9]]
        vb = [["SENATE", 100, 180, .9], ["FUND", 185, 240, .9], ["WGTY-FM", 705, 800, .9]]
        f, _ = X.extract_fields(page("ORDER WORKSHEET\nAdvertiser Station Market\nSENATE FUND WGTY-FM", boxes=[[["ORDER", 10, 90, .9]], hb, vb]), "OCR")
        self.assertEqual(val(f, "supplier_name"), "WGTY-FM")                        # value under a Station column header, aligned by position

    def test_call_sign_letterhead_is_accepted(self):
        self.assertEqual(val(fx("KIT-AM\nINVOICE\nInvoice # 644823-1")[0], "supplier_name"), "KIT-AM")


if __name__ == "__main__":
    unittest.main()
